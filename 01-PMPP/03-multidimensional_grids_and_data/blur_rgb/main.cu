#include <cuda_runtime.h>     // cudaMalloc, cudaMemcpy
#include <iostream>           // cout
#include <opencv2/opencv.hpp> // imread, imwrite

#define CUDA_CHECK(err)                                                                                                \
    if ((err) != cudaSuccess) {                                                                                        \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE);                                                                                            \
    }

__global__ void blur_rgb(uint8_t *bgr, uint8_t *blurred_bgr, int width, int height, int channels, int blur_radius) {
    // 3x3, blur_radius = 1
    // 5x5, blur_radius = 2
    // 7x7, blur_radius = 3

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    int bgr_offset = (y * width + x) * channels * sizeof(uint8_t);

    int b_val = 0;
    int g_val = 0;
    int r_val = 0;
    int pixel_cnt = 0;
    for (int iy = -blur_radius; iy <= blur_radius; ++iy) {
        for (int ix = -blur_radius; ix <= blur_radius; ++ix) {
            int curr_x = x + ix;
            int curr_y = y + iy;

            int curr_bgr_offset = (curr_y * width + curr_x) * channels * sizeof(uint8_t);
            uint8_t b = bgr[curr_bgr_offset + 0];
            uint8_t g = bgr[curr_bgr_offset + 1];
            uint8_t r = bgr[curr_bgr_offset + 2];

            if (curr_x >= 0 && curr_x < width && curr_y >= 0 && curr_y < height) {
                b_val += b;
                g_val += g;
                r_val += r;
                pixel_cnt += 1;
            }
        }
    }

    blurred_bgr[bgr_offset + 0] = static_cast<uint8_t>(b_val / pixel_cnt);
    blurred_bgr[bgr_offset + 1] = static_cast<uint8_t>(g_val / pixel_cnt);
    blurred_bgr[bgr_offset + 2] = static_cast<uint8_t>(r_val / pixel_cnt);
}

// rm -rf build && cmake -B build -S . && cmake --build build && ./build/bin/blur_rgb
int main() {
    // Read image
    cv::Mat image = cv::imread("starry_night.jpeg");
    if (image.empty()) {
        std::cerr << "Error: Could not read image" << std::endl;
        return 1;
    }
    int height = image.rows;
    int width = image.cols;
    int channels = image.channels();
    if (width * channels != image.step[0]) {
        std::cerr << "Error: Width and step[0] do not match" << std::endl;
        return 1;
    }

    std::cout << "Image size: " << height << "x" << width << "x" << channels << std::endl;
    std::cout << "Image type: " << image.type() << std::endl; // CV_8UC3

    uint8_t *d_bgr = nullptr, *d_blurred_bgr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bgr, height * width * channels * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_blurred_bgr, height * width * channels * sizeof(uint8_t)));

    CUDA_CHECK(cudaMemcpy(d_bgr, image.data, height * width * channels * sizeof(uint8_t), cudaMemcpyHostToDevice));

    // Convert to grayscale
    dim3 threads_per_block(16, 16, 1);
    dim3 blocks_per_grid((width + threads_per_block.x - 1) / threads_per_block.x,
                         (height + threads_per_block.y - 1) / threads_per_block.y);
    blur_rgb<<<blocks_per_grid, threads_per_block>>>(d_bgr, d_blurred_bgr, width, height, channels, 2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cv::Mat blurred_image(height, width, CV_8UC3);
    CUDA_CHECK(cudaMemcpy(blurred_image.data, d_blurred_bgr, height * width * channels * sizeof(uint8_t),
                          cudaMemcpyDeviceToHost));

    cv::imwrite("blurred.png", blurred_image);

    // Clean up device memory
    CUDA_CHECK(cudaFree(d_bgr));
    CUDA_CHECK(cudaFree(d_blurred_bgr));

    return 0;
}
