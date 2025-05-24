/**
 * It implements the KLT tracker with CUDA from scratch. It's based on the implementation in
 * tracker_basic.cpp.
 *
 * It tries to follow the points below in the implementation:
 * 1. Encapsulate the reused memory in the class
 * 2. Pre-allocate the required memory in the constructor.
 * 3. For each point, use one cuda stream to track
 * 4. Use cuda library to replace cv.
 * 5. Use async version of cuda functions such that everything can be run asynchronously.
 *
 */

#include <cassert>
#include <chrono>
#include <cmath>
#include <cublas_v2.h>
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

__global__ void getRectSubPixKernel(const float *__restrict__ img, int W, int H, float center_x, float center_y,
                                    float *__restrict__ patch, int win) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= win * win)
        return;

    int px = tid % win;
    int py = tid / win;

    // compute source coords
    float fx = center_x - (win - 1) * 0.5f + px;
    float fy = center_y - (win - 1) * 0.5f + py;

    // replicate border
    fx = fminf(fmaxf(fx, 0.0f), W - 1.0f);
    fy = fminf(fmaxf(fy, 0.0f), H - 1.0f);

    int x0 = int(floorf(fx)), y0 = int(floorf(fy));
    int x1 = min(x0 + 1, W - 1), y1 = min(y0 + 1, H - 1);
    float dx = fx - x0, dy = fy - y0;

    // fetch four neighbors
    float v00 = img[y0 * W + x0];
    float v10 = img[y0 * W + x1];
    float v01 = img[y1 * W + x0];
    float v11 = img[y1 * W + x1];

    // bilinear interpolate
    float v0 = v00 + (v10 - v00) * dx;
    float v1 = v01 + (v11 - v01) * dx;
    patch[py * win + px] = v0 + (v1 - v0) * dy;
}

__host__ void cudaGetRectSubPix(const float *img, int win_size, float center_x, float center_y, float *patch, int width,
                                int height, cudaStream_t stream) {
    int N = win_size * win_size;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    getRectSubPixKernel<<<blocks, threads, 0, stream>>>(img, width, height, center_x, center_y, patch, win_size);

    cudaStreamSynchronize(stream);
}

__host__ void cudaVecSub(const Npp32f *vec1, const Npp32f *vec2, Npp32f *error, int win_size, cudaStream_t stream) {

    NppStreamContext ctx;
    NPP_CHECK(nppGetStreamContext(&ctx));
    ctx.hStream = stream;

    int step = win_size * sizeof(Npp32f);
    NppiSize roi = {win_size, win_size};

    NPP_CHECK(nppiSub_32f_C1R_Ctx(
        /* pSrc1       */ vec1,
        /* nSrc1Step   */ step,
        /* pSrc2       */ vec2,
        /* nSrc2Step   */ step,
        /* pDst        */ error,
        /* nDstStep    */ step,
        /* oSizeROI    */ roi,
        /* nppStreamCtx*/ ctx));
}

// vec1, vec2: device pointers to win_size×win_size floats
// win_size:   side length
// stream:     CUDA stream to enqueue on
__host__ float cudaComputeDot(const float *vec1, const float *vec2, int win_size, cudaStream_t stream,
                              cublasHandle_t cublasH) {
    int N = win_size * win_size;
    float result = 0.0f;

    // 1) tell cuBLAS to use your stream
    cublasSetStream(cublasH, stream);

    // 2) compute dot-product: result = vec1⋅vec2
    //    increments = 1 because they're contiguous
    cublasSdot(cublasH, N, vec1, 1, vec2, 1, &result);

    return result; // this value is valid immediately after the call, async on stream
}

__host__ std::tuple<float, float, bool> lucas_kanade(const Npp8u *d_img1, const Npp8u *d_img2, const Npp32f *d_grad_x,
                                                     const Npp32f *d_grad_y, Npp32f *d_template_patch, Npp32f *d_patch,
                                                     Npp32f *d_patch_grad_x, Npp32f *d_patch_grad_y, float u, float v,
                                                     float x_l, float y_l, float scaled_win_size, int origin_win_size,
                                                     int max_iter, float eps, float min_eig, int width, int height,
                                                     cudaStream_t stream, cublasHandle_t cublasH) {

    NppStreamContext ctx;
    NPP_CHECK(nppGetStreamContext(&ctx));
    ctx.hStream = stream;

    // Convert Npp8u to Npp32f
    Npp32f *d_img1_32f, *d_img2_32f;
    CUDA_CHECK(cudaMalloc(&d_img1_32f, width * height * sizeof(Npp32f)));
    CUDA_CHECK(cudaMalloc(&d_img2_32f, width * height * sizeof(Npp32f)));
    NPP_CHECK(nppiConvert_8u32f_C1R_Ctx(d_img1, width * sizeof(Npp8u), d_img1_32f, width * sizeof(Npp32f),
                                        {width, height}, ctx));
    NPP_CHECK(nppiConvert_8u32f_C1R_Ctx(d_img2, width * sizeof(Npp8u), d_img2_32f, width * sizeof(Npp32f),
                                        {width, height}, ctx));

    // Extract the template patch from I1 (no motion yet)
    cudaGetRectSubPix(d_img1_32f, origin_win_size, x_l, y_l, d_template_patch, width, height, stream);
    // DEBUG
    // save_device_image<Npp32f, CV_32FC1>(d_template_patch, origin_win_size, origin_win_size, "template_patch.png");

    bool success = true;

    // pre-allocate memory for d_error
    Npp32f *d_error;
    CUDA_CHECK(cudaMalloc(&d_error, origin_win_size * origin_win_size * sizeof(Npp32f)));

    for (int iter = 0; iter < max_iter; iter++) {

        float xc = x_l + u;
        float yc = y_l + v;

        // Check if the shifted patch is inside I2
        if (!in_bound(xc, yc, int(scaled_win_size / 2), width, height)) {
            // std::cout << "lucas_kanade Out of bound at iteration " << iter << std::endl;
            success = false;
            break;
        }

        // Sample the patch in I2 and the gradients at (xc, yc)
        cudaGetRectSubPix(d_img2_32f, origin_win_size, xc, yc, d_patch, width, height, stream);
        // DEBUG
        // save_device_image<Npp32f, CV_32FC1>(d_patch, origin_win_size, origin_win_size, "patch.png");

        cudaGetRectSubPix(d_grad_x, origin_win_size, xc, yc, d_patch_grad_x, width, height, stream);
        cudaGetRectSubPix(d_grad_y, origin_win_size, xc, yc, d_patch_grad_y, width, height, stream);

        // Compute error image (template - patch)
        cudaVecSub(d_patch, d_template_patch, d_error, origin_win_size, stream);

        // Build elements of the normal equations matrix
        float Gxx = cudaComputeDot(d_patch_grad_x, d_patch_grad_x, origin_win_size, stream, cublasH);
        float Gxy = cudaComputeDot(d_patch_grad_x, d_patch_grad_y, origin_win_size, stream, cublasH);
        float Gyy = cudaComputeDot(d_patch_grad_y, d_patch_grad_y, origin_win_size, stream, cublasH);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Check for degeneracy and compute minimum eigenvalue
        float det = Gxx * Gyy - Gxy * Gxy;

        if (det <= 0) {
            // std::cout << "lucas_kanade det <= 0 at iteration " << iter << std::endl;
            success = false;
            break;
        }
        float trace = Gxx + Gyy;
        float lambda_min = (trace - std::sqrt(trace * trace - 4 * det)) / 2.0;
        // Filter by minimum eigenvalue (normalized by patch size)
        if (lambda_min / (origin_win_size * origin_win_size) < min_eig) {
            // std::cout << "lucas_kanade lambda_min < min_eig at iteration " << iter << std::endl;
            success = false;
            break;
        }

        // Compute right-hand side vector
        float b1 = cudaComputeDot(d_patch_grad_x, d_error, origin_win_size, stream, cublasH);
        float b2 = cudaComputeDot(d_patch_grad_y, d_error, origin_win_size, stream, cublasH);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Solve for [du, dv] using Cramer's rule
        float inv_det = 1.0f / det;
        float du = (Gyy * b1 - Gxy * b2) * inv_det;
        float dv = (-Gxy * b1 + Gxx * b2) * inv_det;

        u += du;
        v += dv;

        // Check convergence
        if (std::abs(du) < eps && std::abs(dv) < eps) {
            break;
        }
    }

    CUDA_CHECK(cudaFree(d_img1_32f));
    CUDA_CHECK(cudaFree(d_img2_32f));
    CUDA_CHECK(cudaFree(d_error));

    return std::make_tuple(u, v, success);
}

__host__ std::tuple<float, float, bool>
pyramid_lucas_kanade(std::vector<Npp8u *> &d_pyr1, std::vector<Npp8u *> &d_pyr2, std::vector<Npp32f *> &d_grad_x,
                     std::vector<Npp32f *> &d_grad_y, Npp32f *d_template_patch, Npp32f *d_patch, Npp32f *d_patch_grad_x,
                     Npp32f *d_patch_grad_y, const cv::Point2f &pt, int levels, int win_size, int max_iter, float eps,
                     float min_eig, int width, int height, cudaStream_t stream, cublasHandle_t cublasH) {
    float u_prev = 0.0;
    float v_prev = 0.0;
    bool success = true;

    for (int lvl = levels; lvl >= 0; lvl--) { // 3, 2, 1, 0
        int pyr_width = width >> lvl;
        int pyr_height = height >> lvl;

        float scale = 1.0f / (1 << lvl);

        float x_l = pt.x * scale;
        float y_l = pt.y * scale;

        float u = u_prev * 2.0;
        float v = v_prev * 2.0;

        // Check if the patch around (x_l, y_l) is inside I1
        if (!in_bound(x_l, y_l, int(scale * win_size / 2), pyr_width, pyr_height)) {
            // std::cout << "Out of bound at level " << lvl << std::endl;
            success = false;
            break;
        }

        auto [new_u, new_v, success] =
            lucas_kanade(d_pyr1[lvl], d_pyr2[lvl], d_grad_x[lvl], d_grad_y[lvl], d_template_patch, d_patch,
                         d_patch_grad_x, d_patch_grad_y, u, v, x_l, y_l, scale * win_size, win_size, max_iter, eps,
                         min_eig, pyr_width, pyr_height, stream, cublasH);

        if (!success) {
            // std::cout << "Failed to converge at level " << lvl << std::endl;
            break;
        }

        // Save refined flow for this level
        u_prev = new_u;
        v_prev = new_v;
    }

    return std::make_tuple(u_prev, v_prev, success);
}

__host__ void build_pyramid(std::vector<Npp8u *> &pyr, int levels, const int width, const int height,
                            cudaStream_t stream) {

    NppStreamContext ctx;
    NPP_CHECK(nppGetStreamContext(&ctx));
    ctx.hStream = stream;

    for (int i = 0; i < levels; i++) { // 0, 1, 2
        int src_w = width >> i;
        int src_h = height >> i;
        int dst_w = width >> (i + 1);
        int dst_h = height >> (i + 1);
        int srcStep = src_w * sizeof(Npp8u);
        int dstStep = dst_w * sizeof(Npp8u);

        NppiSize srcSize = {src_w, src_h};
        NppiRect srcROI = {0, 0, src_w, src_h};
        NppiSize dstSize = {dst_w, dst_h};
        NppiRect dstROI = {0, 0, dst_w, dst_h};
        NPP_CHECK(nppiResize_8u_C1R_Ctx(
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

        // OpenCV works. But I don't want to use it.
        // cv::cuda::GpuMat d_src(src_h, src_w, CV_8UC1, pyr[i]);
        // cv::cuda::GpuMat d_dst(dst_h, dst_w, CV_8UC1, pyr[i + 1]);
        // cv::cuda::Stream cvStream = cv::cuda::StreamAccessor::wrapStream(stream);
        // cv::cuda::pyrDown(d_src, d_dst, cvStream);

        // NPP GaussPyramidLayerDown does not work. I don't know why.
        // NppiSize srcSize = {src_w, src_h};
        // NppiPoint srcOffset = {0, 0};
        // NppiSize dstSize = {dst_w, dst_h};
        // static const Npp32f gauss5f[5] = {1.0f / 16.0f, 4.0f / 16.0f, 6.0f / 16.0f, 4.0f / 16.0f, 1.0f / 16.0f};
        // NPP_CHECK(nppiFilterGaussPyramidLayerDownBorder_8u_C1R_Ctx(
        //     /* pSrc           */ pyr[i],
        //     /* nSrcStep       */ srcStep,
        //     /* oSrcSize       */ srcSize,
        //     /* oSrcOffset     */ srcOffset,
        //     /* pDst           */ pyr[i + 1],
        //     /* nDstStep       */ dstStep,
        //     /* oSizeROI       */ dstSize,
        //     /* nRate          */ 2.0f,    // downsample rate (2.0 = keep every 2nd pixel)
        //     /* nFilterTaps    */ 5,       // length of gauss5f[]
        //     /* pKernel        */ gauss5f, // normalized float kernel
        //     /* eBorderType    */ NPP_BORDER_REPLICATE,
        //     /* nppStreamCtx   */ ctx));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

__host__ void apply_sobel_filter(Npp8u *src, Npp32f *grad_x, Npp32f *grad_y, int width, int height,
                                 cudaStream_t stream) {

    NppStreamContext ctx;
    NPP_CHECK(nppGetStreamContext(&ctx));
    ctx.hStream = stream;

    Npp32f *d_temp = nullptr;
    size_t elems = size_t(width) * size_t(height);
    CUDA_CHECK(cudaMalloc(&d_temp, elems * sizeof(Npp32f)));

    int srcStep = width * sizeof(Npp8u);
    int dstStep = width * sizeof(Npp32f);
    NppiSize imgSize = {width, height};
    NppiPoint imgOffset = {0, 0};
    NppiSize roiSize = {width, height};

    NPP_CHECK(nppiConvert_8u32f_C1R_Ctx(src, srcStep, d_temp, dstStep, roiSize, ctx));

    // grad_x = ∂/∂x → use the *Vert* filter in NPP
    NPP_CHECK(nppiFilterSobelVertBorder_32f_C1R_Ctx(d_temp, dstStep, imgSize, imgOffset, grad_x, dstStep, roiSize,
                                                    NPP_BORDER_REPLICATE, ctx));

    // grad_y = ∂/∂y → use the *Horiz* filter in NPP
    NPP_CHECK(nppiFilterSobelHorizBorder_32f_C1R_Ctx(d_temp, dstStep, imgSize, imgOffset, grad_y, dstStep, roiSize,
                                                     NPP_BORDER_REPLICATE, ctx));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_temp));
}

class SparseOpticalFlow {
  private:
    int levels;
    int height, width;
    std::vector<cv::Point2f> prev_pts;
    std::vector<cv::Point2f> next_pts;

    // GPU memory
    Npp8u *d_bgr, *d_gray;

    std::vector<Npp8u *> d_pyr1, d_pyr2;
    std::vector<Npp32f *> d_grad_x, d_grad_y;

    Npp32f *d_template_patch, *d_patch;
    Npp32f *d_patch_grad_x, *d_patch_grad_y;

    // CUDA streams for parallel processing
    std::vector<cudaStream_t> streams;

    int win_size;
    int max_iter;
    float eps;
    float min_eig;

    cublasHandle_t cublasH;

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
    assert(width == init_gray.step[0]);

    // Initialize CUDA streams
    streams.resize(pts.size());
    for (auto &stream : streams) {
        CUDA_CHECK(cudaStreamCreate(&stream));
    }

    cublasCreate(&cublasH);

    size_t bgrBytes = size_t(height) * width * 3u; // 3 channels
    size_t grayBytes = size_t(height) * width;
    CUDA_CHECK(cudaMalloc(&d_bgr, bgrBytes));
    CUDA_CHECK(cudaMalloc(&d_gray, grayBytes));

    // Initialize pyramid vectors
    d_pyr1.resize(levels + 1);
    d_pyr2.resize(levels + 1);
    d_grad_x.resize(levels + 1);
    d_grad_y.resize(levels + 1);

    // Allocate GPU memory for pyramid, d_pyr1 and d_pyr2
    for (int i = 0; i <= levels; i++) { // 0, 1, 2, 3
        // divide by 2^i
        const int pyr_width = width >> i;
        const int pyr_height = height >> i;

        size_t pyr_size = pyr_width * pyr_height * sizeof(Npp8u);
        CUDA_CHECK(cudaMalloc((void **)&d_pyr1[i], pyr_size));
        CUDA_CHECK(cudaMalloc((void **)&d_pyr2[i], pyr_size));
    }
    // Build initial pyramid d_pyr1
    CUDA_CHECK(cudaMemcpy(d_pyr1[0], init_gray.data, grayBytes, cudaMemcpyHostToDevice));
    build_pyramid(d_pyr1, levels, width, height, streams[0]);
    // DEBUG
    // for (int i = 0; i <= levels; i++) {
    //     const int pyr_width = width >> i;
    //     const int pyr_height = height >> i;
    //     save_device_image<Npp8u, CV_8UC1>(d_pyr1[i], pyr_width, pyr_height, "pyr1_" + std::to_string(i) + ".png");
    // }

    // Allocate GPU memory for d_grad_x and d_grad_y
    for (int i = 0; i <= levels; i++) { // 0, 1, 2, 3
        // divide by 2^i
        const int pyr_width = width >> i;
        const int pyr_height = height >> i;

        size_t grad_size = pyr_width * pyr_height * sizeof(Npp32f);
        CUDA_CHECK(cudaMalloc(&d_grad_x[i], grad_size));
        CUDA_CHECK(cudaMalloc(&d_grad_y[i], grad_size));
    }

    size_t patch_size = win_size * win_size * sizeof(Npp32f);
    CUDA_CHECK(cudaMalloc(&d_template_patch, patch_size));
    CUDA_CHECK(cudaMalloc(&d_patch, patch_size));
    CUDA_CHECK(cudaMalloc(&d_patch_grad_x, patch_size));
    CUDA_CHECK(cudaMalloc(&d_patch_grad_y, patch_size));

    prev_pts = pts;
    next_pts = std::vector<cv::Point2f>(pts.size(), cv::Point2f(0, 0));
}

__global__ void bgr_to_gray(const unsigned char *src, unsigned char *dst, size_t srcStep, size_t dstStep, int width,
                            int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    const unsigned char *rowSrc = src + y * srcStep;
    unsigned char *rowDst = dst + y * dstStep;

    // BGR channels are interleaved
    const float b = rowSrc[3 * x + 0];
    const float g = rowSrc[3 * x + 1];
    const float r = rowSrc[3 * x + 2];

    float gray = __fmaf_rn(r, 0.299f, __fmaf_rn(g, 0.587f, __fmul_rn(b, 0.114f)));
    rowDst[x] = (unsigned char)(gray + 0.5f);
}

__host__ void apply_bgr_to_gray(Npp8u *src, Npp8u *dst, size_t srcStep, size_t dstStep, int width, int height) {
    dim3 block(32, 32, 1);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);
    bgr_to_gray<<<grid, block, 0>>>(src, dst, srcStep, dstStep, width, height);
}

__host__ std::vector<cv::Point2f> SparseOpticalFlow::track(cv::Mat &next_bgr) {
    int width = next_bgr.cols;
    int height = next_bgr.rows;
    size_t bgrStepBytes = next_bgr.step[0];
    assert(bgrStepBytes == width * 3u);
    size_t grayStepBytes = width * sizeof(Npp8u);

    /**
     * Convert BGR to grayscale
     * next_bgr is on pageable memory
     */
    CUDA_CHECK(cudaMemcpy(d_bgr, next_bgr.data, bgrStepBytes * height, cudaMemcpyHostToDevice));
    apply_bgr_to_gray(d_bgr, d_gray, bgrStepBytes, grayStepBytes, width, height);
    // DEBUG
    // save_device_image<Npp8u, CV_8UC1>(d_gray, width, height, "d_gray.png");

    // Build pyramid pyr2
    CUDA_CHECK(cudaMemcpy(d_pyr2[0], d_gray, grayStepBytes * height, cudaMemcpyDeviceToDevice));
    build_pyramid(this->d_pyr2, levels, width, height, streams[0]);
    // DEBUG
    // for (int i = 0; i <= levels; i++) {
    //     const int pyr_width = width >> i;
    //     const int pyr_height = height >> i;
    //     save_device_image<Npp8u, CV_8UC1>(d_pyr2[i], pyr_width, pyr_height, "pyr2_" + std::to_string(i) + ".png");
    // }

    CUDA_CHECK(cudaStreamSynchronize(streams[0]));

    // Pre-compute gradients for all pyramid levels
    for (int i = 0; i <= levels; i++) { // 0, 1, 2, 3
        const int pyr_width = width >> i;
        const int pyr_height = height >> i;
        apply_sobel_filter(this->d_pyr2[i], this->d_grad_x[i], this->d_grad_y[i], pyr_width, pyr_height, streams[0]);
    }
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));
    // DEBUG
    // for (int i = 0; i <= levels; i++) {
    //     const int pyr_width = width >> i;
    //     const int pyr_height = height >> i;
    //     save_device_image<Npp32f, CV_32FC1>(d_grad_x[i], pyr_width, pyr_height,
    //                                         "d_grad_x_" + std::to_string(i) + ".png");
    //     save_device_image<Npp32f, CV_32FC1>(d_grad_y[i], pyr_width, pyr_height,
    //                                         "d_grad_y_" + std::to_string(i) + ".png");
    // }

    // Track each point in parallel using separate CUDA streams
    for (size_t i = 0; i < prev_pts.size(); ++i) {
        float x = prev_pts[i].x;
        float y = prev_pts[i].y;

        // Pyramidal Lucas-Kanade
        auto [u, v, success] =
            pyramid_lucas_kanade(this->d_pyr1, this->d_pyr2, this->d_grad_x, this->d_grad_y, this->d_template_patch,
                                 this->d_patch, this->d_patch_grad_x, this->d_patch_grad_y, prev_pts[i], levels,
                                 win_size, max_iter, eps, min_eig, this->width, this->height, streams[i], cublasH);

        if (success) {
            next_pts[i].x = x + u;
            next_pts[i].y = y + v;
        } else {
            next_pts[i].x = x;
            next_pts[i].y = y;
        }
    }

    // Synchronize all streams
    for (auto &stream : streams) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // swap d_pyr1 and d_pyr2
    std::swap(d_pyr1, d_pyr2);

    prev_pts = next_pts;

    return next_pts;
}

SparseOpticalFlow::~SparseOpticalFlow() {
    // Free GPU memory
    CUDA_CHECK(cudaFree(d_bgr));
    CUDA_CHECK(cudaFree(d_gray));

    for (auto &d_pyr : d_pyr1) {
        CUDA_CHECK(cudaFree(d_pyr));
    }
    for (auto &d_pyr : d_pyr2) {
        CUDA_CHECK(cudaFree(d_pyr));
    }

    for (auto &d_grad : d_grad_x) {
        CUDA_CHECK(cudaFree(d_grad));
    }
    for (auto &d_grad : d_grad_y) {
        CUDA_CHECK(cudaFree(d_grad));
    }

    CUDA_CHECK(cudaFree(d_template_patch));
    CUDA_CHECK(cudaFree(d_patch));
    CUDA_CHECK(cudaFree(d_patch_grad_x));
    CUDA_CHECK(cudaFree(d_patch_grad_y));

    // Destroy CUDA streams
    for (auto &stream : streams) {
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    cublasDestroy(cublasH);
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
    std::string input_mp4 = "data/1920x1080_30fps_8s.mp4";
    std::string output_mp4 = "output/tracker_cuda_naive.mp4";

    std::vector<cv::Point2f> prev_pts;
    prev_pts.push_back(cv::Point2f(1676, 654));
    prev_pts.push_back(cv::Point2f(1740, 699));
    prev_pts.push_back(cv::Point2f(1825, 690));

    std::vector<cv::Point2f> next_pts;

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
    cv::Mat next_bgr;
    while (true) {
        if (!cap.read(next_bgr))
            break;

        auto t1 = std::chrono::high_resolution_clock::now(); // start time

        std::vector<cv::Point2f> next_pts = sof.track(next_bgr);

        auto t2 = std::chrono::high_resolution_clock::now(); // end time
        accum_time += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        for (size_t i = 0; i < prev_pts.size(); ++i) {
            trajectory[i].push_back(next_pts[i]);
        }

        // Uncomment to visualize the result
        // cv::Mat display = next_bgr.clone();
        // plot_trajectory(display, trajectory);
        // writer.write(display);
    }

    std::cout << "Total time of all frames: " << accum_time << " microseconds. "
              << "Each frame average time: " << accum_time / total_frames << " microseconds." << std::endl;

    cap.release();
    writer.release();
    return 0;
}
