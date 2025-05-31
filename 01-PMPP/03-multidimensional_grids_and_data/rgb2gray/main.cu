#include <cuda_runtime.h>     // cudaMalloc, cudaMemcpy
#include <iostream>           // cout
#include <opencv2/opencv.hpp> // imread, imwrite

#define CUDA_CHECK(err)                                                                                                \
    if ((err) != cudaSuccess) {                                                                                        \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE);                                                                                            \
    }

__global__ void bgr2gray(uint8_t *bgr, uint8_t *gray, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    int gray_offset = y * width + x;
    int bgr_offset = gray_offset * channels;
    uint8_t b = bgr[bgr_offset + 0];
    uint8_t g = bgr[bgr_offset + 1];
    uint8_t r = bgr[bgr_offset + 2];

    gray[gray_offset] = 0.299 * r + 0.587 * g + 0.114 * b;
}

// rm -rf build && cmake -B build -S . && cmake --build build && ./build/bin/rgb2gray
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

    uint8_t *d_bgr = nullptr, *d_gray = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bgr, height * width * channels * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_gray, height * width * sizeof(uint8_t)));

    CUDA_CHECK(cudaMemcpy(d_bgr, image.data, height * width * channels * sizeof(uint8_t), cudaMemcpyHostToDevice));

    // Convert to grayscale
    dim3 threads_per_block(16, 16, 1);
    dim3 blocks_per_grid((width + threads_per_block.x - 1) / threads_per_block.x,
                         (height + threads_per_block.y - 1) / threads_per_block.y);
    bgr2gray<<<blocks_per_grid, threads_per_block>>>(d_bgr, d_gray, width, height, channels);

    cv::Mat gray_image(height, width, CV_8UC1);
    CUDA_CHECK(cudaMemcpy(gray_image.data, d_gray, height * width * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    cv::imwrite("gray.png", gray_image);

    return 0;
}
