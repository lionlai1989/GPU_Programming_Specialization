#include <cuda_runtime.h>
#include <getopt.h>
#include <math_constants.h>
#include <npp.h>
#include <nppi.h>
#include <nppi_filtering_functions.h>

#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>

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

void apply_sobel_filter(Npp8u *d_src, size_t srcStepBytes, int width, int height, Npp16s *d_magnitude,
                        Npp32f *d_direction, int kernel_size, cudaStream_t stream) {
    Npp16s *d_gradient_x = nullptr;
    Npp16s *d_gradient_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_gradient_x, height * width * sizeof(Npp16s)));
    CUDA_CHECK(cudaMalloc(&d_gradient_y, height * width * sizeof(Npp16s)));

    NppiSize roi = {width, height};
    NppiPoint oSrcOffset = {0, 0};

    NppStreamContext ctx;
    ctx.hStream = stream;
    NppStatus status = nppiGradientVectorSobelBorder_8u16s_C1R_Ctx(d_src,                  // pSrc
                                                                   srcStepBytes,           // nSrcStep
                                                                   roi,                    // oSrcSize
                                                                   oSrcOffset,             // oSrcOffset
                                                                   d_gradient_x,           // pDstX
                                                                   width * sizeof(Npp16s), // nDstXStep
                                                                   d_gradient_y,           // pDstY
                                                                   width * sizeof(Npp16s), // nDstYStep
                                                                   d_magnitude,            // pDstMag
                                                                   width * sizeof(Npp16s), // nDstMagStep
                                                                   d_direction,            // pDstAngle
                                                                   width * sizeof(Npp32f), // nDstAngleStep
                                                                   roi,                    // oSizeRO
                                                                   NPP_MASK_SIZE_3_X_3,    // eMaskSize
                                                                   nppiNormL2,             // eNorm
                                                                   NPP_BORDER_REPLICATE,   // eBorderType
                                                                   ctx);
    NPP_CHECK(status);

    CUDA_CHECK(cudaFree(d_gradient_x));
    CUDA_CHECK(cudaFree(d_gradient_y));
}

__global__ void non_max_suppression_kernel(Npp16s *suppress, const Npp16s *mag, const Npp32f *dir, size_t magStep,
                                           size_t angStep, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
        if (x < width && y < height) {
            suppress[y * (magStep / sizeof(Npp16s)) + x] = 0;
        }
        return;
    }

    int magStride = magStep / sizeof(Npp16s);
    int angStride = angStep / sizeof(Npp32f);
    Npp16s m = mag[y * magStride + x];                       // central mag
    float a = dir[y * angStride + x] * 180.0f / CUDART_PI_F; // rad to deg
    if (a < 0.0f)
        a += 180.0f;

    // Quantize angle to one of 4 directions
    int dx1, dy1, dx2, dy2;
    if ((a < 22.5f) || (a >= 157.5f)) {
        dx1 = -1;
        dy1 = 0;
        dx2 = 1;
        dy2 = 0; // 0: left/right
    } else if (a < 67.5f) {
        dx1 = -1;
        dy1 = 1;
        dx2 = 1;
        dy2 = -1; // 45: bottom‑left/top‑right
    } else if (a < 112.5f) {
        dx1 = 0;
        dy1 = 1;
        dx2 = 0;
        dy2 = -1; // 90: up/down
    } else {
        dx1 = -1;
        dy1 = -1;
        dx2 = 1;
        dy2 = 1; // 135: top‑left/bottom‑right
    }

    // Compare to neighbors
    Npp16s m1 = mag[(y + dy1) * magStride + (x + dx1)];
    Npp16s m2 = mag[(y + dy2) * magStride + (x + dx2)];
    suppress[y * magStride + x] = (m >= m1 && m >= m2) ? m : 0;
}

void apply_non_max_suppression(Npp16s *d_suppress, Npp16s *d_magnitude, Npp32f *d_direction, size_t magStep,
                               size_t angStep, int width, int height, cudaStream_t stream) {
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    non_max_suppression_kernel<<<grid, block, 0, stream>>>(d_suppress, d_magnitude, d_direction, magStep, angStep,
                                                           width, height);

    CUDA_CHECK(cudaGetLastError());
}

__global__ void hysteresis_thresholding_kernel(Npp8u *edge_map, const Npp16s *mag, size_t magStep, int width,
                                               int height, int low_th, int high_th) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int magStride = magStep / sizeof(Npp16s);
    if (x >= width || y >= height)
        return;
    int idx = y * magStride + x;

    Npp16s m = mag[idx];

    if (m >= high_th) {
        edge_map[y * width + x] = 255;
        return;
    }
    if (m < low_th) {
        edge_map[y * width + x] = 0;
        return;
    }

    // weak edge: check 8‐neighbors for any strong magnitude
    bool connected = false;
    // loop over neighborhood
    for (int dy = -1; dy <= 1 && !connected; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0)
                continue;
            int nx = x + dx, ny = y + dy;
            if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                continue;
            Npp16s m2 = mag[ny * magStride + nx];
            if (m2 >= high_th) {
                connected = true;
                break;
            }
        }
    }
    edge_map[y * width + x] = connected ? 255 : 0;
}

void apply_hysteresis_thresholding(Npp8u *d_edge_map, const Npp16s *d_suppress_mag, int width, int height,
                                   int low_threshold, int high_threshold, cudaStream_t stream) {
    CUDA_CHECK(cudaMemset(d_edge_map, 0, width * height * sizeof(Npp8u)));

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    hysteresis_thresholding_kernel<<<grid, block, 0, stream>>>(d_edge_map, d_suppress_mag, width * sizeof(Npp16s),
                                                               width, height, low_threshold, high_threshold);

    CUDA_CHECK(cudaGetLastError());
}

__global__ void bgr_to_gray(const unsigned char *src, unsigned char *dst, size_t srcPitch, size_t dstPitch, int width,
                            int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    // Row pointers using pitched memory
    const unsigned char *rowSrc = src + y * srcPitch;
    unsigned char *rowDst = dst + y * dstPitch;

    // BGR channels are interleaved
    int idx = 3 * x;
    float b = rowSrc[idx + 0];
    float g = rowSrc[idx + 1];
    float r = rowSrc[idx + 2];

    float gray = __fmaf_rn(r, 0.299f, __fmaf_rn(g, 0.587f, __fmul_rn(b, 0.114f)));
    rowDst[x] = (unsigned char)(gray + 0.5f);
}

void apply_bgr_to_gray(Npp8u *src, Npp8u *dst, size_t srcStep, size_t dstStep, int width, int height,
                       cudaStream_t stream) {
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    bgr_to_gray<<<grid, block, 0, stream>>>(src, dst, srcStep, dstStep, width, height);
}

void apply_gaussian_filter(Npp8u *d_gray, Npp8u *d_blur, size_t srcStep, size_t dstStep, int width, int height,
                           cudaStream_t stream) {
    // Define ROI (whole image) and offsets
    NppiSize roiSize = {width, height};
    NppiPoint srcOffset = {0, 0};

    // NPP supports mask sizes {3,5,7,9,11,13,15}
    const NppiMaskSize mask = NPP_MASK_SIZE_3_X_3;

    // Apply Gaussian filter with border replication
    NppStreamContext ctx;
    ctx.hStream = stream;
    NppStatus status = nppiFilterGaussBorder_8u_C1R_Ctx(
        /* pSrc         */ d_gray,
        /* nSrcStep     */ srcStep,
        /* oSrcSize     */ roiSize,
        /* oSrcOffset   */ srcOffset,
        /* pDst         */ d_blur,
        /* nDstStep     */ dstStep,
        /* oSizeROI     */ roiSize,
        /* eMaskSize    */ mask,
        /* eBorderType  */ NPP_BORDER_REPLICATE,
        /* nppStreamCtx */ ctx);
    NPP_CHECK(status);
}

void process_image(const cv::Mat &input_bgr, cv::Mat &output_gray, cudaStream_t stream) {

    int width = input_bgr.cols;
    int height = input_bgr.rows;
    size_t bgrStepBytes = input_bgr.step[0];
    size_t gryStepBytes = width * sizeof(Npp8u);

    Npp8u *d_bgr = nullptr;

    // Allocate and copy BGR to device
    CUDA_CHECK(cudaMalloc(&d_bgr, height * bgrStepBytes));
    CUDA_CHECK(cudaMemcpy(d_bgr, input_bgr.data, height * bgrStepBytes, cudaMemcpyHostToDevice));

    // Allocate grayscale buffers
    Npp8u *d_gray = nullptr, *d_blur = nullptr;
    CUDA_CHECK(cudaMalloc(&d_gray, height * gryStepBytes));
    CUDA_CHECK(cudaMalloc(&d_blur, height * gryStepBytes));

    // BGR to Gray conversion
    apply_bgr_to_gray(d_bgr, d_gray, bgrStepBytes, gryStepBytes, width, height, stream);

    // Gaussian blur
    apply_gaussian_filter(d_gray, d_blur, gryStepBytes, gryStepBytes, width, height, stream);

    // Sobel filter
    Npp16s *d_magnitude = nullptr;
    Npp32f *d_direction = nullptr;
    size_t magStep = width * sizeof(Npp16s);
    size_t angStep = width * sizeof(Npp32f);
    CUDA_CHECK(cudaMalloc(&d_magnitude, height * magStep));
    CUDA_CHECK(cudaMalloc(&d_direction, height * angStep));
    apply_sobel_filter(d_blur, gryStepBytes, width, height, d_magnitude, d_direction, 3, stream);

    // Non-max suppression
    Npp16s *d_suppress_magnitude = nullptr;
    CUDA_CHECK(cudaMalloc(&d_suppress_magnitude, height * magStep));
    apply_non_max_suppression(d_suppress_magnitude, d_magnitude, d_direction, magStep, angStep, width, height, stream);

    // Hysteresis thresholding
    Npp8u *d_edge_map = nullptr;
    CUDA_CHECK(cudaMalloc(&d_edge_map, height * width * sizeof(Npp8u)));
    apply_hysteresis_thresholding(d_edge_map, d_suppress_magnitude, width, height, 25, 100, stream);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(output_gray.data, d_edge_map, height * output_gray.step[0], cudaMemcpyDeviceToHost));

    // Cleanup device memory
    CUDA_CHECK(cudaFree(d_bgr));
    CUDA_CHECK(cudaFree(d_gray));
    CUDA_CHECK(cudaFree(d_blur));
    CUDA_CHECK(cudaFree(d_magnitude));
    CUDA_CHECK(cudaFree(d_direction));
    CUDA_CHECK(cudaFree(d_suppress_magnitude));
    CUDA_CHECK(cudaFree(d_edge_map));
}

class BlockingQueue {
  private:
    std::queue<cv::Mat> q;
    std::mutex mtx;
    std::condition_variable cv_not_empty;
    std::condition_variable cv_not_full;
    bool is_done = false;
    const size_t max_size = 300; // Maximum number of frames to buffer

  public:
    void enqueue(cv::Mat &&frame) { // Use move semantics
        std::unique_lock<std::mutex> lock(mtx);
        cv_not_full.wait(lock, [this]() { return q.size() < max_size || is_done; });
        if (!is_done) {
            q.push(std::move(frame)); // Move the frame instead of copying
            cv_not_empty.notify_one();
        }
    }

    bool dequeue(cv::Mat &frame) {
        std::unique_lock<std::mutex> lock(mtx);
        cv_not_empty.wait(lock, [this]() { return !q.empty() || is_done; });
        if (q.empty() && is_done) {
            return false;
        }
        frame = std::move(q.front()); // Move the frame instead of copying
        q.pop();
        cv_not_full.notify_one();
        return true;
    }

    void set_done() {
        std::unique_lock<std::mutex> lock(mtx);
        is_done = true;
        cv_not_empty.notify_all();
        cv_not_full.notify_all();
    }
};

void reader_thread(BlockingQueue &queue, cv::VideoCapture &cap) {
    cv::Mat frame;
    while (cap.read(frame)) {
        queue.enqueue(std::move(frame));
    }
    queue.set_done();
}

void writer_thread(BlockingQueue &queue, cv::VideoWriter &writer) {
    cv::Mat frame;
    while (queue.dequeue(frame)) {
        writer.write(frame);
    }
}

int main(int argc, char *argv[]) {
    std::string input_file{"data/1920x960_100sec_30fps.mp4"}; // 3840x1920_100sec_30fps
    std::string output_file{"canny_cuda_naive.mp4"};

    if (input_file.empty() || output_file.empty()) {
        exit(EXIT_FAILURE);
    }

    cv::VideoCapture cap(input_file, cv::CAP_FFMPEG);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video file: " << input_file << "\n";
        exit(EXIT_FAILURE);
    }

    // Get video properties
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    // Create video writer with H.264 codec
    cv::VideoWriter writer(output_file, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), // H.264 codec
                           fps, cv::Size(width, height),
                           false); // false for grayscale output

    if (!writer.isOpened()) {
        std::cerr << "Error: Cannot create video writer for: " << output_file << "\n";
        exit(EXIT_FAILURE);
    }

    std::cout << "Processing video: " << width << "x" << height << " @ " << fps << " fps" << " with " << total_frames
              << " frames" << std::endl;
    std::cout << "Start timing ..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    BlockingQueue reader_queue;
    BlockingQueue writer_queue;
    std::thread reader_thr(reader_thread, std::ref(reader_queue), std::ref(cap));
    std::thread writer_thr(writer_thread, std::ref(writer_queue), std::ref(writer));

    cudaStream_t stream_[1];
    CUDA_CHECK(cudaStreamCreate(&stream_[0]));

    // Process each frame
    cv::Mat frame;
    long long accum_time = 0;
    int frame_count = 0;
    while (reader_queue.dequeue(frame)) {
        auto t1 = std::chrono::high_resolution_clock::now();

        cv::Mat edge_map(height, width, CV_8UC1); // Must be instantiated here
        process_image(frame, edge_map, stream_[0]);
        CUDA_CHECK(cudaStreamSynchronize(stream_[0]));
        frame_count += 1;

        auto t2 = std::chrono::high_resolution_clock::now();
        accum_time += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        writer_queue.enqueue(std::move(edge_map));
    }
    assert(frame_count == total_frames);

    std::cout << "Total time of all frame: " << accum_time
              << " microseconds. Each frame average time: " << accum_time / total_frames << " microseconds."
              << std::endl;

    reader_queue.set_done();
    writer_queue.set_done();
    reader_thr.join();
    writer_thr.join();

    // Release resources
    writer.release();
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for compute device to finish.
    CUDA_CHECK(cudaDeviceReset());

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "End-to-end runtime: " << duration.count() << " milliseconds" << std::endl;

    std::cout << "Done. Output saved as " << output_file << std::endl;
    return 0;
}
