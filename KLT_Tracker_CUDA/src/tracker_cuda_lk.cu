/**
 * tracker_cuda_lk.cu implements the KLT tracker with CUDA from scratch. It aims to outperform all other
 * implementations.
 */

#include <cassert>
#include <chrono>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <npp.h>
#include <nppcore.h>
#include <nppi.h>
#include <nppi_arithmetic_and_logical_operations.h>
#include <nppi_data_exchange_and_initialization.h>
#include <nppi_filtering_functions.h>
#include <nppi_geometry_transforms.h>
#include <nppi_statistics_functions.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <tuple>
#include <vector>

#define CUDA_CHECK(err)                                                                                                \
    if ((err) != cudaSuccess) {                                                                                        \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE);                                                                                            \
    }

#define NPP_CHECK(err)                                                                                                 \
    if ((err) != NPP_SUCCESS) {                                                                                        \
        std::cerr << "NPP error at " << __FILE__ << ":" << __LINE__ << " - status = " << err << std::endl;             \
        exit(EXIT_FAILURE);                                                                                            \
    }

template <typename T, int CV_TYPE>
__host__ void save_device_image(const T *d_ptr, int width, int height, const std::string &filename) {
    cv::Mat h_mat(height, width, CV_TYPE);
    CUDA_CHECK(cudaMemcpy(h_mat.data, d_ptr, size_t(width) * height * sizeof(T), cudaMemcpyDeviceToHost));
    cv::imwrite(filename, h_mat);
}

__host__ __device__ static inline bool in_bound(const float x, const float y, const int half_win, const int width,
                                                const int height) {
    return x >= half_win && x < width - half_win && y >= half_win && y < height - half_win;
}

// Device function: bilinear interpolation in a single-channel image
__device__ float bilinear_interpolate(const float *img, int width, int height, float x, float y) {
    int x0 = floorf(x), y0 = floorf(y);
    int x1 = x0 + 1, y1 = y0 + 1;
    // Clamp to image bounds
    x0 = max(0, min(x0, width - 1));
    y0 = max(0, min(y0, height - 1));
    x1 = max(0, min(x1, width - 1));
    y1 = max(0, min(y1, height - 1));
    float dx = x - x0;
    float dy = y - y0;
    // Fetch four neighbors
    float I00 = img[y0 * width + x0];
    float I10 = img[y0 * width + x1];
    float I01 = img[y1 * width + x0];
    float I11 = img[y1 * width + x1];
    // Interpolate along x then y
    float I0 = I00 * (1.0f - dx) + I10 * dx;
    float I1 = I01 * (1.0f - dx) + I11 * dx;
    return I0 * (1.0f - dy) + I1 * dy;
}

// Optimized Lucas-Kanade kernel running on device
__device__ void lucas_kanade_kernel(const float *image0, const float *image1, const float *grad_x, const float *grad_y,
                                    int width, int height, float x0, float y0, int patch_radius, int max_iter,
                                    float eps, float min_eigenvalue, float u_in, float v_in, float *u_out,
                                    float *v_out) {
    int patch_size = 2 * patch_radius + 1;
    int patch_area = patch_size * patch_size;
    // FIX: The kernel is launched with 1D blocks (threads_per_block is a scalar).
    // So threadIdx.y is always 0. We must compute tx, ty from the 1D thread index.
    int local_idx = threadIdx.x;
    int tx = local_idx % patch_size;
    int ty = local_idx / patch_size;
    // Note: We assume threads_per_block == patch_area. All threads must participate
    // in __syncthreads() calls, so no early return here.

    // Quick boundary check for the entire patch
    if (x0 < patch_radius || x0 > width - 1 - patch_radius || y0 < patch_radius || y0 > height - 1 - patch_radius) {
        // Output input guess if out of bounds
        *u_out = u_in;
        *v_out = v_in;
        return;
    }

    // Allocate shared memory: reference patch + partial sums + shared u,v
    // Layout:
    // s_I0: patch_area
    // s_Gxx: patch_area
    // s_Gxy: patch_area
    // s_Gyy: patch_area
    // s_b1:  patch_area
    // s_b2:  patch_area
    // s_uv:  2 floats
    extern __shared__ float sdata[];
    float *s_I0 = sdata;              // patch_area floats
    float *s_Gxx = s_I0 + patch_area; // patch_area floats
    float *s_Gxy = s_Gxx + patch_area;
    float *s_Gyy = s_Gxy + patch_area;
    float *s_b1 = s_Gyy + patch_area;
    float *s_b2 = s_b1 + patch_area;

    // Shared variables for broadcasting displacement and exit flag
    // We can use the end of s_b2 or specific slots.
    // Let's use the space after s_b2.
    volatile float *s_common = s_b2 + patch_area;
    // s_common[0] = u, s_common[1] = v, s_common[2] = break_flag

    // Load reference patch from image0 into shared memory
    float xr = x0 - patch_radius + tx;
    float yr = y0 - patch_radius + ty;
    s_I0[local_idx] = bilinear_interpolate(image0, width, height, xr, yr);

    // Initialize shared displacement
    if (local_idx == 0) {
        s_common[0] = u_in;
        s_common[1] = v_in;
        s_common[2] = 0.0f; // break flag
    }
    __syncthreads();

    // Iterative Lucas-Kanade refinement
    for (int iter = 0; iter < max_iter; ++iter) {
        // Read current u, v from shared memory
        float u = s_common[0];
        float v = s_common[1];

        // Compute current sample position in image1
        float xc = xr + u;
        float yc = yr + v;

        // Check if out-of-bounds (we need +1/-1 for gradient)
        // If any thread is out of bounds, we might have issues.
        // We just clamp or check individual validity.
        // For reduction correctness, we should set invalid contributions to 0.
        bool valid = (xc >= 1.0f && xc <= width - 2 && yc >= 1.0f && yc <= height - 2);

        float gx = 0.0f, gy = 0.0f, e = 0.0f;
        if (valid) {
            // Sample I1
            float I1_val = bilinear_interpolate(image1, width, height, xc, yc);
            // Use precomputed gradients for speed
            gx = bilinear_interpolate(grad_x, width, height, xc, yc);
            gy = bilinear_interpolate(grad_y, width, height, xc, yc);
            // Compute error = I0 - I1
            e = s_I0[local_idx] - I1_val;
        }

        // Store per-pixel contributions in shared arrays
        // If invalid, contribute 0
        s_Gxx[local_idx] = valid ? gx * gx : 0.0f;
        s_Gxy[local_idx] = valid ? gx * gy : 0.0f;
        s_Gyy[local_idx] = valid ? gy * gy : 0.0f;
        s_b1[local_idx] = valid ? gx * e : 0.0f;
        s_b2[local_idx] = valid ? gy * e : 0.0f;
        __syncthreads();

        // Parallel Reduction
        // Reduce s_Gxx, s_Gxy, s_Gyy, s_b1, s_b2
        // Assuming block dimensions handle patch_area.
        // We use a stride loop for arbitrary size.
        for (unsigned int s = 512; s > 0; s >>= 1) {
            if (local_idx < s) {
                if (local_idx + s < patch_area) {
                    s_Gxx[local_idx] += s_Gxx[local_idx + s];
                    s_Gxy[local_idx] += s_Gxy[local_idx + s];
                    s_Gyy[local_idx] += s_Gyy[local_idx + s];
                    s_b1[local_idx] += s_b1[local_idx + s];
                    s_b2[local_idx] += s_b2[local_idx + s];
                }
            }
            __syncthreads();
        }

        // Solve 2x2 system on thread 0
        if (local_idx == 0) {
            float sumGxx = s_Gxx[0];
            float sumGxy = s_Gxy[0];
            float sumGyy = s_Gyy[0];
            float sumb1 = s_b1[0];
            float sumb2 = s_b2[0];

            float det = sumGxx * sumGyy - sumGxy * sumGxy;
            float du = 0.0f, dv = 0.0f;

            if (fabs(det) < 1e-6f) {
                // Ill-conditioned; terminate iteration
                s_common[2] = 1.0f; // Set break flag
            } else {
                float inv00 = sumGyy / det;
                float inv01 = -sumGxy / det;
                float inv11 = sumGxx / det;
                du = inv00 * sumb1 + inv01 * sumb2;
                dv = inv01 * sumb1 + inv11 * sumb2;

                // Update u, v
                s_common[0] += du;
                s_common[1] += dv;

                // Check convergence
                if (du * du + dv * dv < eps * eps) {
                    s_common[2] = 1.0f; // Converged
                }

                // Optional: Check min eigenvalue
                /*
                float trace = sumGxx + sumGyy;
                float temp = sqrtf((sumGxx - sumGyy) * (sumGxx - sumGyy) + 4.0f * sumGxy * sumGxy);
                float lambda2 = 0.5f * (trace - temp);
                if (lambda2 < min_eigenvalue) {
                    s_common[2] = 1.0f;
                }
                */
            }
        }
        __syncthreads();

        // Check break flag
        if (s_common[2] > 0.0f)
            break;
    }

    // Write output displacement (ALL threads need to write to their local variables
    // to propagate to the next level of the pyramid)
    // Read final u,v from shared memory
    *u_out = s_common[0];
    *v_out = s_common[1];
}

// Use __launch_bounds__ to tell the compiler to optimize for this thread count
// This helps reduce register usage per thread
__global__ void __launch_bounds__(1024, 1)
    pyramid_lucas_kanade_kernel(const cv::Point2f *prev_pts, cv::Point2f *next_pts, const float **d_pyr1_ptrs,
                                const float **d_pyr2_ptrs, const float **d_grad_x_ptrs, const float **d_grad_y_ptrs,
                                int levels, int win_size, int max_iter, float eps, float min_eig, int width, int height,
                                int num_points) {
    int pt_idx = blockIdx.x;
    if (pt_idx >= num_points)
        return;

    float x_pt = prev_pts[pt_idx].x;
    float y_pt = prev_pts[pt_idx].y;

    float u_prev = 0.0f;
    float v_prev = 0.0f;

    for (int lvl = levels; lvl >= 0; lvl--) { // 3, 2, 1, 0
        int pyr_width = width >> lvl;
        int pyr_height = height >> lvl;

        float scale = 1.0f / (1 << lvl);
        float x_l = x_pt * scale;
        float y_l = y_pt * scale;

        float u_in = u_prev * 2.0f;
        float v_in = v_prev * 2.0f;

        // Check if the patch around (x_l, y_l) is inside bounds
        if (!in_bound(x_l, y_l, win_size / 2, pyr_width, pyr_height)) {
            // Keep u_prev, v_prev as is (likely 0 or scaled)
            // Or reset? OpenCV usually resets or keeps.
            // If we break, u_prev/v_prev won't be updated for lower levels.
            // We should probably just zero it or stop.
            // If we break here, u_prev is from higher level (smaller image).
            // We need to scale it up for next levels?
            // Actually, if we lose track at coarse level, we probably can't recover at fine level.
            // But we must maintain consistency.
            u_prev *= 2.0f;
            v_prev *= 2.0f;
            continue;
        }

        float u_out, v_out;
        lucas_kanade_kernel(d_pyr1_ptrs[lvl], d_pyr2_ptrs[lvl], d_grad_x_ptrs[lvl], d_grad_y_ptrs[lvl], pyr_width,
                            pyr_height, x_l, y_l, win_size / 2, max_iter, eps, min_eig, u_in, v_in, &u_out, &v_out);

        u_prev = u_out;
        v_prev = v_out;
    }

    // Write back result (only thread 0 writes to avoid redundancy)
    if (threadIdx.x == 0) {
        next_pts[pt_idx].x = x_pt + u_prev;
        next_pts[pt_idx].y = y_pt + v_prev;
    }
}

__host__ void build_pyramid(std::vector<Npp32f *> &pyr, int levels, const int width, const int height,
                            cudaStream_t stream) {

    NppStreamContext ctx;
    NPP_CHECK(nppGetStreamContext(&ctx));
    ctx.hStream = stream;

    for (int i = 0; i < levels; i++) { // 0, 1, 2
        int src_w = width >> i;
        int src_h = height >> i;
        int dst_w = width >> (i + 1);
        int dst_h = height >> (i + 1);
        int srcStep = src_w * sizeof(Npp32f);
        int dstStep = dst_w * sizeof(Npp32f);

        NppiSize srcSize = {src_w, src_h};
        NppiRect srcROI = {0, 0, src_w, src_h};
        NppiSize dstSize = {dst_w, dst_h};
        NppiRect dstROI = {0, 0, dst_w, dst_h};
        NPP_CHECK(nppiResize_32f_C1R_Ctx(
            /* pSrc         */ pyr[i],
            /* nSrcStep     */ srcStep,
            /* oSrcSize     */ srcSize,
            /* oSrcROI      */ srcROI,
            /* pDst         */ pyr[i + 1],
            /* nDstStep     */ dstStep,
            /* oDstSize     */ dstSize,
            /* oDstROI      */ dstROI,
            /* eInterpolation */ NPPI_INTER_LINEAR,
            /* nppStreamCtx */ ctx));
    }
}

__host__ void apply_sobel_filter(Npp32f *src, Npp32f *grad_x, Npp32f *grad_y, int width, int height,
                                 cudaStream_t stream) {

    NppStreamContext ctx;
    NPP_CHECK(nppGetStreamContext(&ctx));
    ctx.hStream = stream;

    int srcStep = width * sizeof(Npp32f);
    int dstStep = width * sizeof(Npp32f);
    NppiSize imgSize = {width, height};
    NppiPoint imgOffset = {0, 0};
    NppiSize roiSize = {width, height};

    // "Vert" finds vertical edges (gradient in x direction)
    NPP_CHECK(nppiFilterSobelVertBorder_32f_C1R_Ctx(src, srcStep, imgSize, imgOffset, grad_x, dstStep, roiSize,
                                                    NPP_BORDER_REPLICATE, ctx));

    // "Horiz" finds horizontal edges (gradient in y direction)
    NPP_CHECK(nppiFilterSobelHorizBorder_32f_C1R_Ctx(src, srcStep, imgSize, imgOffset, grad_y, dstStep, roiSize,
                                                     NPP_BORDER_REPLICATE, ctx));
}

class SparseOpticalFlow {
  private:
    int levels;
    int height, width;
    std::vector<cv::Point2f> prev_pts;

    // GPU memory
    Npp8u *d_bgr, *d_gray;

    // Vectors to manage memory allocations (Host vector of Device pointers)
    std::vector<Npp32f *> d_pyr1, d_pyr2;
    std::vector<Npp32f *> d_grad_x, d_grad_y;

    // Device pointers to pointer arrays (Device pointer to Device pointers)
    Npp32f **d_pyr1_ptrs, **d_pyr2_ptrs;
    Npp32f **d_grad_x_ptrs, **d_grad_y_ptrs;

    // Device memory for points
    cv::Point2f *d_prev_pts, *d_next_pts;

    cudaStream_t stream; // Main stream
    NppStreamContext main_npp_ctx;

    int win_size;
    int max_iter;
    float eps;
    float min_eig;

  public:
    SparseOpticalFlow(std::vector<cv::Point2f> &pts, cv::Mat &init_gray, int levels = 3, int win_size = 31,
                      int max_iter = 30, float eps = 0.01, float min_eig = 1e-4);
    ~SparseOpticalFlow();

    std::vector<cv::Point2f> track(cv::Mat &next_bgr);
};

SparseOpticalFlow::SparseOpticalFlow(std::vector<cv::Point2f> &pts, cv::Mat &init_gray, int levels, int win_size,
                                     int max_iter, float eps, float min_eig)
    : levels(levels), win_size(win_size), max_iter(max_iter), eps(eps), min_eig(min_eig) {

    height = init_gray.rows;
    width = init_gray.cols;

    // Create main stream
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Initialize NPP context
    NPP_CHECK(nppGetStreamContext(&main_npp_ctx));
    main_npp_ctx.hStream = stream;

    size_t bgrBytes = size_t(height) * width * 3u; // 3 channels
    size_t grayBytes = size_t(height) * width;
    CUDA_CHECK(cudaMallocAsync(&d_bgr, bgrBytes, stream));
    CUDA_CHECK(cudaMallocAsync(&d_gray, grayBytes, stream));

    // Initialize pyramid vectors
    d_pyr1.resize(levels + 1);
    d_pyr2.resize(levels + 1);
    d_grad_x.resize(levels + 1);
    d_grad_y.resize(levels + 1);

    // Allocate GPU memory for pyramid levels
    for (int i = 0; i <= levels; i++) {
        const int pyr_width = width >> i;
        const int pyr_height = height >> i;
        size_t pyr_size = pyr_width * pyr_height * sizeof(Npp32f);
        CUDA_CHECK(cudaMallocAsync(&d_pyr1[i], pyr_size, stream));
        CUDA_CHECK(cudaMallocAsync(&d_pyr2[i], pyr_size, stream));
        CUDA_CHECK(cudaMallocAsync(&d_grad_x[i], pyr_size, stream));
        CUDA_CHECK(cudaMallocAsync(&d_grad_y[i], pyr_size, stream));
    }

    // Allocate device arrays for pointers
    CUDA_CHECK(cudaMallocAsync(&d_pyr1_ptrs, (levels + 1) * sizeof(Npp32f *), stream));
    CUDA_CHECK(cudaMallocAsync(&d_pyr2_ptrs, (levels + 1) * sizeof(Npp32f *), stream));
    CUDA_CHECK(cudaMallocAsync(&d_grad_x_ptrs, (levels + 1) * sizeof(Npp32f *), stream));
    CUDA_CHECK(cudaMallocAsync(&d_grad_y_ptrs, (levels + 1) * sizeof(Npp32f *), stream));

    // Copy pointers to device
    // Note: d_pyr vectors contain device pointers. We copy this array of pointers to the device.
    CUDA_CHECK(
        cudaMemcpyAsync(d_pyr1_ptrs, d_pyr1.data(), (levels + 1) * sizeof(Npp32f *), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_pyr2_ptrs, d_pyr2.data(), (levels + 1) * sizeof(Npp32f *), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_grad_x_ptrs, d_grad_x.data(), (levels + 1) * sizeof(Npp32f *), cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(d_grad_y_ptrs, d_grad_y.data(), (levels + 1) * sizeof(Npp32f *), cudaMemcpyHostToDevice,
                               stream));

    // Build initial pyramid d_pyr1
    CUDA_CHECK(cudaMemcpyAsync(d_gray, init_gray.data, grayBytes, cudaMemcpyHostToDevice, stream));
    NPP_CHECK(nppiConvert_8u32f_C1R_Ctx(d_gray, width * sizeof(Npp8u), d_pyr1[0], width * sizeof(Npp32f),
                                        {width, height}, main_npp_ctx));
    build_pyramid(d_pyr1, levels, width, height, stream);

    // Initialize points
    prev_pts = pts;
    size_t pts_bytes = pts.size() * sizeof(cv::Point2f);
    CUDA_CHECK(cudaMallocAsync(&d_prev_pts, pts_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&d_next_pts, pts_bytes, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_prev_pts, pts.data(), pts_bytes, cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

__global__ void bgr_to_gray(const unsigned char *src, unsigned char *dst, size_t srcStep, size_t dstStep, int width,
                            int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    const unsigned char *rowSrc = src + y * srcStep;
    unsigned char *rowDst = dst + y * dstStep;

    const float b = rowSrc[3 * x + 0];
    const float g = rowSrc[3 * x + 1];
    const float r = rowSrc[3 * x + 2];

    float gray = __fmaf_rn(r, 0.299f, __fmaf_rn(g, 0.587f, __fmul_rn(b, 0.114f)));
    rowDst[x] = (unsigned char)(gray + 0.5f);
}

__host__ void apply_bgr_to_gray(Npp8u *src, Npp8u *dst, size_t srcStep, size_t dstStep, int width, int height,
                                cudaStream_t stream) {
    dim3 block(32, 32, 1);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);
    bgr_to_gray<<<grid, block, 0, stream>>>(src, dst, srcStep, dstStep, width, height);
}

__host__ std::vector<cv::Point2f> SparseOpticalFlow::track(cv::Mat &next_bgr) {
    int width = next_bgr.cols;
    int height = next_bgr.rows;
    size_t bgrStepBytes = next_bgr.step[0];
    size_t grayStepBytes = width * sizeof(Npp8u);

    // Upload and convert
    CUDA_CHECK(cudaMemcpyAsync(d_bgr, next_bgr.data, bgrStepBytes * height, cudaMemcpyHostToDevice, stream));
    apply_bgr_to_gray(d_bgr, d_gray, bgrStepBytes, grayStepBytes, width, height, stream);

    // Build pyramid pyr2
    NPP_CHECK(nppiConvert_8u32f_C1R_Ctx(d_gray, width * sizeof(Npp8u), d_pyr2[0], width * sizeof(Npp32f),
                                        {width, height}, main_npp_ctx));
    build_pyramid(this->d_pyr2, levels, width, height, stream);

    // Pre-compute gradients for all pyramid levels of pyr2 (Image I1)
    for (int i = 0; i <= levels; i++) {
        const int pyr_width = width >> i;
        const int pyr_height = height >> i;
        apply_sobel_filter(this->d_pyr2[i], this->d_grad_x[i], this->d_grad_y[i], pyr_width, pyr_height, stream);
    }

    // Update device pointer arrays (d_pyr1 and d_pyr2 might have been swapped on host)
    // We only need to update the pointers to the image data.
    // NOTE: std::swap(d_pyr1, d_pyr2) swaps the vectors, so d_pyr1[i] now points to what was d_pyr2[i].
    // So we need to update d_pyr1_ptrs and d_pyr2_ptrs on device with new values.
    CUDA_CHECK(
        cudaMemcpyAsync(d_pyr1_ptrs, d_pyr1.data(), (levels + 1) * sizeof(Npp32f *), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_pyr2_ptrs, d_pyr2.data(), (levels + 1) * sizeof(Npp32f *), cudaMemcpyHostToDevice, stream));

    // Gradients d_grad_x/y are computed on d_pyr2. So we update them too (though they don't swap, they are
    // overwritten).
    CUDA_CHECK(cudaMemcpyAsync(d_grad_x_ptrs, d_grad_x.data(), (levels + 1) * sizeof(Npp32f *), cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(d_grad_y_ptrs, d_grad_y.data(), (levels + 1) * sizeof(Npp32f *), cudaMemcpyHostToDevice,
                               stream));

    // Launch batched kernel
    // 1 block per point.
    // Threads: win_size * win_size.
    int threads_per_block = win_size * win_size;

    // Shared memory size: 6 arrays of size patch_area * sizeof(float) + extra for shared vars
    // 6 * patch_area + some extra.
    size_t shared_mem = (6 * threads_per_block + 32) * sizeof(float);

    pyramid_lucas_kanade_kernel<<<prev_pts.size(), threads_per_block, shared_mem, stream>>>(
        d_prev_pts, d_next_pts, (const float **)d_pyr1_ptrs, (const float **)d_pyr2_ptrs, (const float **)d_grad_x_ptrs,
        (const float **)d_grad_y_ptrs, levels, win_size, max_iter, eps, min_eig, width, height, prev_pts.size());
    // Check for kernel launch errors (capture error before it's cleared)
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // Download results
    std::vector<cv::Point2f> next_pts(prev_pts.size());
    CUDA_CHECK(cudaMemcpyAsync(next_pts.data(), d_next_pts, prev_pts.size() * sizeof(cv::Point2f),
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Prepare for next frame
    std::swap(d_pyr1, d_pyr2);
    // d_prev_pts for next frame is d_next_pts of this frame.
    // We can just swap pointers on device? No, d_prev_pts is a single pointer.
    // But we need to update the data in d_prev_pts for the next iteration.
    // Since we computed new points in d_next_pts, we can swap d_prev_pts and d_next_pts pointers?
    // BUT we didn't allocate them as "frame 1" and "frame 2" pointers that persist.
    // Actually we can just swap the pointers member variables.
    std::swap(d_prev_pts, d_next_pts);

    // Also update host prev_pts
    prev_pts = next_pts;

    return next_pts;
}

SparseOpticalFlow::~SparseOpticalFlow() {
    // Free GPU memory
    CUDA_CHECK(cudaFreeAsync(d_bgr, stream));
    CUDA_CHECK(cudaFreeAsync(d_gray, stream));

    for (auto &d_pyr : d_pyr1)
        CUDA_CHECK(cudaFreeAsync(d_pyr, stream));
    for (auto &d_pyr : d_pyr2)
        CUDA_CHECK(cudaFreeAsync(d_pyr, stream));
    for (auto &d_grad : d_grad_x)
        CUDA_CHECK(cudaFreeAsync(d_grad, stream));
    for (auto &d_grad : d_grad_y)
        CUDA_CHECK(cudaFreeAsync(d_grad, stream));

    CUDA_CHECK(cudaFreeAsync(d_pyr1_ptrs, stream));
    CUDA_CHECK(cudaFreeAsync(d_pyr2_ptrs, stream));
    CUDA_CHECK(cudaFreeAsync(d_grad_x_ptrs, stream));
    CUDA_CHECK(cudaFreeAsync(d_grad_y_ptrs, stream));

    CUDA_CHECK(cudaFreeAsync(d_prev_pts, stream));
    CUDA_CHECK(cudaFreeAsync(d_next_pts, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void plot_trajectory(cv::Mat &display, const std::vector<std::vector<cv::Point2f>> &trajectory) {
    std::vector<cv::Scalar> line_color = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255)};
    std::vector<cv::Scalar> point_color = {cv::Scalar(0, 0, 255), cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0)};

    for (size_t i = 0; i < trajectory.size(); ++i) {
        for (size_t j = 1; j < trajectory[i].size(); ++j) {
            cv::line(display, trajectory[i][j - 1], trajectory[i][j], line_color[i], 2);
        }

        cv::circle(display, trajectory[i].back(), 5, point_color[i], -1);
    }
}

int main(int argc, char **argv) {
    // Use pinned host memory for cv::Mat
    cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator(cv::cuda::HostMem::AllocType::PAGE_LOCKED));

    std::string input_mp4 = "data/1920x1080_30fps_8s.mp4";
    std::string output_mp4 = "output/tracker_cuda_lk.mp4";

    std::vector<cv::Point2f> prev_pts;
    prev_pts.push_back(cv::Point2f(1676, 654));
    prev_pts.push_back(cv::Point2f(1740, 699));
    prev_pts.push_back(cv::Point2f(1825, 690));

    // Store all tracked points for visualization
    std::vector<std::vector<cv::Point2f>> trajectory(prev_pts.size());

    // Initialize tracked points with starting positions
    for (size_t i = 0; i < prev_pts.size(); ++i) {
        trajectory[i].push_back(prev_pts[i]);
    }

    cv::VideoCapture cap(input_mp4);
    if (!cap.isOpened()) {
        std::cout << "Error: Could not open video file" << std::endl;
        return -1;
    }
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    std::cout << "Total frames: " << total_frames << std::endl;

    cv::Mat prev_bgr, prev_gray;
    cap.read(prev_bgr);
    if (prev_bgr.empty()) {
        std::cout << "Error: Could not read first frame" << std::endl;
        return -1;
    }
    cv::cvtColor(prev_bgr, prev_gray, cv::COLOR_BGR2GRAY);
    int height = prev_bgr.rows;
    int width = prev_bgr.cols;

    cv::VideoWriter writer(output_mp4, cv::VideoWriter::fourcc('M', 'P', '4', 'V'), 30, cv::Size(width, height));

    // Init SparseOpticalFlow
    SparseOpticalFlow sof(prev_pts, prev_gray);

    long long accum_time = 0;

    cv::cuda::HostMem h_frame(height, width, CV_8UC3, cv::cuda::HostMem::AllocType::PAGE_LOCKED);
    cv::Mat next_bgr = h_frame.createMatHeader();
    while (true) {
        if (!cap.read(next_bgr))
            break;

        auto t1 = std::chrono::high_resolution_clock::now(); // start time

        std::vector<cv::Point2f> next_pts = sof.track(next_bgr);

        auto t2 = std::chrono::high_resolution_clock::now(); // end time
        accum_time += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        // Uncomment to visualize the result
        // for (size_t i = 0; i < prev_pts.size(); ++i) {
        //     trajectory[i].push_back(next_pts[i]);
        // }
        // cv::Mat display = next_bgr.clone();
        // plot_trajectory(display, trajectory);
        // writer.write(display);
    }

    std::cout << "Total time of all frames: " << accum_time << " microseconds. "
              << "Each frame average time: " << accum_time / total_frames << " microseconds." << std::endl;

    cap.release();
    writer.release();
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for compute device to finish.

    return 0;
}
