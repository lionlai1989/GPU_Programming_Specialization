#include <cuda_runtime.h>
#include <getopt.h>
#include <math_constants.h>
#include <npp.h>
#include <nppi.h>
#include <nppi_filtering_functions.h>

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>

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

void get_normalized_kernel(Npp32s *normalized_kernel, const Npp32s target_sum) {
    constexpr int kernelSize = 11;
    constexpr Npp32s kernel[kernelSize * kernelSize] = {
        1,  4,   13,  25,   36,   40,   36,   25,   13,  4,   1,  // \n
        4,  16,  52,  100,  144,  160,  144,  100,  52,  16,  4,  // \n
        13, 52,  169, 325,  468,  520,  468,  325,  169, 52,  13, // \n
        25, 100, 325, 625,  900,  1000, 900,  625,  325, 100, 25, // \n
        36, 144, 468, 900,  1296, 1440, 1296, 900,  468, 144, 36, // \n
        40, 160, 520, 1000, 1440, 1600, 1440, 1000, 520, 160, 40, // \n
        36, 144, 468, 900,  1296, 1440, 1296, 900,  468, 144, 36, // \n
        25, 100, 325, 625,  900,  1000, 900,  625,  325, 100, 25, // \n
        13, 52,  169, 325,  468,  520,  468,  325,  169, 52,  13, // \n
        4,  16,  52,  100,  144,  160,  144,  100,  52,  16,  4,  // \n
        1,  4,   13,  25,   36,   40,   36,   25,   13,  4,   1   // \n
    };

    Npp32s kernel_sum = 0;
    for (int i = 0; i < kernelSize * kernelSize; ++i) {
        kernel_sum += kernel[i];
    }

    for (int i = 0; i < kernelSize * kernelSize; ++i) {
        normalized_kernel[i] = kernel[i] * target_sum / kernel_sum;
    }
}

void apply_sobel_filter(Npp8u *d_src, size_t srcStepBytes, int width, int height, Npp16s *d_magnitude,
                        Npp32f *d_direction, int kernel_size) {
    Npp16s *d_gradient_x = nullptr;
    Npp16s *d_gradient_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_gradient_x, height * width * sizeof(Npp16s)));
    CUDA_CHECK(cudaMalloc(&d_gradient_y, height * width * sizeof(Npp16s)));

    NppiSize roi = {width, height};
    NppiPoint oSrcOffset = {0, 0};

    NppStatus status = nppiGradientVectorSobelBorder_8u16s_C1R(d_src,                  // pSrc
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
                                                               NPP_BORDER_REPLICATE    // eBorderType
    );
    NPP_CHECK(status);

    CUDA_CHECK(cudaFree(d_gradient_x));
    CUDA_CHECK(cudaFree(d_gradient_y));
}

__global__ void non_max_suppression_kernel(Npp16s *suppress, const Npp16s *mag, const Npp32f *dir, size_t magStep,
                                           size_t angStep, int width, int height) {
    // Compute pixel coords
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
        // zero borders
        if (x < width && y < height) {
            suppress[y * (magStep / sizeof(Npp16s)) + x] = 0;
        }
        return;
    }

    // Load magnitude & direction
    int magStride = magStep / sizeof(Npp16s);
    int angStride = angStep / sizeof(Npp32f);
    Npp16s m = mag[y * magStride + x];                       // central mag
    float a = dir[y * angStride + x] * 180.0f / CUDART_PI_F; // rad→deg
    if (a < 0.0f)
        a += 180.0f;

    // Quantize angle to one of 4 directions
    int dx1, dy1, dx2, dy2;
    if ((a < 22.5f) || (a >= 157.5f)) {
        dx1 = -1;
        dy1 = 0;
        dx2 = 1;
        dy2 = 0; // 0°: left/right
    } else if (a < 67.5f) {
        dx1 = -1;
        dy1 = 1;
        dx2 = 1;
        dy2 = -1; // 45°: bottom‑left/top‑right
    } else if (a < 112.5f) {
        dx1 = 0;
        dy1 = 1;
        dx2 = 0;
        dy2 = -1; // 90°: up/down
    } else {
        dx1 = -1;
        dy1 = -1;
        dx2 = 1;
        dy2 = 1; // 135°: top‑left/bottom‑right
    }

    // Compare to neighbors
    Npp16s m1 = mag[(y + dy1) * magStride + (x + dx1)];         // neighbor 1
    Npp16s m2 = mag[(y + dy2) * magStride + (x + dx2)];         // neighbor 2
    suppress[y * magStride + x] = (m >= m1 && m >= m2) ? m : 0; // keep or zero
}

void apply_non_max_suppression(Npp16s *d_suppress, Npp16s *d_magnitude, Npp32f *d_direction, size_t magStep,
                               size_t angStep, int width, int height) {
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Launch kernel
    non_max_suppression_kernel<<<grid, block>>>(d_suppress, d_magnitude, d_direction, magStep, angStep, width, height);

    // Check for errors
    cudaError_t cuErr = cudaGetLastError();
    if (cuErr != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(cuErr) << std::endl;
        exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();
}

__global__ void hysteresis_thresholding_kernel(Npp8u *edge_map, const Npp16s *mag, size_t magStep, int width,
                                               int height, int low_th, int high_th) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // compute 1D indices & strides
    int magStride = magStep / sizeof(Npp16s);
    if (x >= width || y >= height)
        return;
    int idx = y * magStride + x;

    Npp16s m = mag[idx];

    // strong edge?
    if (m >= high_th) {
        edge_map[y * width + x] = 255;
        return;
    }
    // definitely reject?
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
                                   int low_threshold, int high_threshold) {
    // Zero‐initialize edge map
    CUDA_CHECK(cudaMemset(d_edge_map, 0, width * height * sizeof(Npp8u)));

    // Launch parameters
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    hysteresis_thresholding_kernel<<<grid, block>>>(d_edge_map, d_suppress_mag, width * sizeof(Npp16s), width, height,
                                                    low_threshold, high_threshold);

    // Check errors and synchronize
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

cv::Mat process_image(const cv::Mat &input_bgr) {
    // Convert BGR to RGB (NPP expects RGB order)
    cv::Mat input_rgb;
    cv::cvtColor(input_bgr, input_rgb, cv::COLOR_BGR2RGB);

    // Get image dimensions and verify format
    int width = input_rgb.cols;
    int height = input_rgb.rows;
    int channels = input_rgb.channels();
    if (channels != 3) {
        std::cerr << "Error: Input image must be 3-channel RGB" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Allocate device memory
    size_t srcStepBytes = input_rgb.step[0]; // bytes per row (with padding)
    size_t grayStepBytes = width * sizeof(Npp8u);

    Npp8u *d_src = nullptr, *d_gray = nullptr, *d_blur = nullptr;

    // Allocate device memory with error checking
    CUDA_CHECK(cudaMalloc(&d_src, height * srcStepBytes));
    CUDA_CHECK(cudaMalloc(&d_gray, height * grayStepBytes));
    CUDA_CHECK(cudaMalloc(&d_blur, height * grayStepBytes));

    // Initialize output buffers to zero
    CUDA_CHECK(cudaMemset(d_gray, 0, height * grayStepBytes));
    CUDA_CHECK(cudaMemset(d_blur, 0, height * grayStepBytes));

    // Copy RGB image to device
    CUDA_CHECK(cudaMemcpy(d_src, input_rgb.data, height * srcStepBytes, cudaMemcpyHostToDevice));

    // Convert RGB to Grayscale
    NppiSize roiSize = {width, height};
    NppStatus status = nppiRGBToGray_8u_C3C1R(d_src,         // source pointer
                                              srcStepBytes,  // source stride in bytes
                                              d_gray,        // destination pointer
                                              grayStepBytes, // destination stride in bytes
                                              roiSize        // ROI size
    );
    NPP_CHECK(status);

    constexpr int kernelSize = 11;
    constexpr int kernelRadius = kernelSize / 2;
    constexpr Npp32s target_sum = 256;
    Npp32s normalized_kernel[kernelSize * kernelSize];
    get_normalized_kernel(normalized_kernel, target_sum);

    // Allocate device memory for kernel
    Npp32s *d_kernel = nullptr;
    CUDA_CHECK(cudaMalloc(&d_kernel, kernelSize * kernelSize * sizeof(Npp32s)));
    CUDA_CHECK(
        cudaMemcpy(d_kernel, normalized_kernel, kernelSize * kernelSize * sizeof(Npp32s), cudaMemcpyHostToDevice));

    // Apply custom Gaussian filter with border replication
    NppiSize roi = {width, height};
    NppiPoint oSrcOffset = {0, 0};
    NppiPoint oAnchor = {kernelRadius, kernelRadius}; // Center of the kernel
    NppiSize oKernelSize = {kernelSize, kernelSize};

    status = nppiFilterBorder_8u_C1R(d_gray,              // pSrc
                                     grayStepBytes,       // nSrcStep
                                     roi,                 // oSrcSize
                                     oSrcOffset,          // oSrcOffset
                                     d_blur,              // pDst
                                     grayStepBytes,       // nDstStep
                                     roi,                 // oSizeROI
                                     d_kernel,            // pKernel
                                     oKernelSize,         // oKernelSize
                                     oAnchor,             // oAnchor
                                     target_sum,          // nDivisor
                                     NPP_BORDER_REPLICATE // eBorderType
    );
    NPP_CHECK(status);

    Npp16s *d_magnitude = nullptr;
    Npp32f *d_direction = nullptr;
    size_t magStep = width * sizeof(Npp16s);
    size_t angStep = width * sizeof(Npp32f);
    CUDA_CHECK(cudaMalloc(&d_magnitude, height * magStep));
    CUDA_CHECK(cudaMalloc(&d_direction, height * angStep));
    apply_sobel_filter(d_blur, grayStepBytes, width, height, d_magnitude, d_direction, 3);
    /**
     * Debug d_magnitude. Convert 16s to 8u.
     * cv::Mat h_magnitude(height, width, CV_16SC1);
     * CUDA_CHECK(cudaMemcpy(h_magnitude.data, d_magnitude,
     *                       height * h_magnitude.step[0],
     *                       cudaMemcpyDeviceToHost));
     * h_magnitude.convertTo(output_gray, CV_8UC1);
     */

    Npp16s *d_suppress_magnitude = nullptr;
    CUDA_CHECK(cudaMalloc(&d_suppress_magnitude, height * magStep));
    apply_non_max_suppression(d_suppress_magnitude, d_magnitude, d_direction, magStep, angStep, width, height);

    Npp8u *d_edge_map = nullptr;
    CUDA_CHECK(cudaMalloc(&d_edge_map, height * width * sizeof(Npp8u)));
    apply_hysteresis_thresholding(d_edge_map, d_suppress_magnitude, width, height, 25, 100);

    cv::Mat output_gray(height, width, CV_8UC1);

    CUDA_CHECK(cudaMemcpy(output_gray.data, d_edge_map, height * output_gray.step[0], cudaMemcpyDeviceToHost));

    cudaFree(d_src);
    cudaFree(d_gray);
    cudaFree(d_blur);
    cudaFree(d_kernel);
    cudaFree(d_magnitude);
    cudaFree(d_direction);
    cudaFree(d_suppress_magnitude);
    cudaFree(d_edge_map);

    return output_gray;
}

void print_usage(const char *program_name) {
    std::cerr << "Usage: " << program_name << " -i <input_video> -o <output_video>\n"
              << "Options:\n"
              << "  -i <input_video>  Input video file (MP4)\n"
              << "  -o <output_video> Output video file (MP4)\n";
}

int main(int argc, char *argv[]) {
    std::string input_file;
    std::string output_file;

    int opt;
    while ((opt = getopt(argc, argv, "i:o:")) != -1) {
        switch (opt) {
        case 'i':
            input_file = optarg;
            break;
        case 'o':
            output_file = optarg;
            break;
        default:
            print_usage(argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    if (input_file.empty() || output_file.empty()) {
        print_usage(argv[0]);
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

    // Create output directory if it doesn't exist
    std::filesystem::path output_path(output_file);
    std::filesystem::create_directories(output_path.parent_path());

    // Create video writer with H.264 codec
    cv::VideoWriter writer(output_file, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), // H.264 codec
                           fps, cv::Size(width, height),
                           false); // false for grayscale output

    if (!writer.isOpened()) {
        std::cerr << "Error: Cannot create video writer for: " << output_file << "\n";
        exit(EXIT_FAILURE);
    }

    std::cout << "Processing video: " << width << "x" << height << " @ " << fps << " fps" << std::endl;

    // Process each frame
    cv::Mat frame;
    int frame_count = 0;
    while (cap.read(frame)) {
        cv::Mat processed_frame = process_image(frame);

        writer.write(processed_frame);

        frame_count++;
        if (frame_count % 10 == 0) {
            std::cout << "Processed frame " << frame_count << " of " << total_frames << std::endl;
        }
    }

    // Release resources
    cap.release();
    writer.release();

    std::cout << "Done. Output saved as " << output_file << std::endl;
    return 0;
}
