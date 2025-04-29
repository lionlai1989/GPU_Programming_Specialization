#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
#include <string>
#include <tuple>

__device__ __constant__ int d_rows;
__device__ __constant__ int d_columns;

#define CUDA_CHECK(err)                                                                                                \
    if ((err) != cudaSuccess) {                                                                                        \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE);                                                                                            \
    }

__host__ void rgb_to_gray_cpu(const cv::Mat &bgr_img, cv::Mat &gray_img) {
    int width = bgr_img.cols;
    int height = bgr_img.rows;

    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            cv::Vec3b intensity = bgr_img.at<cv::Vec3b>(r, c);
            uchar blue = intensity.val[0];
            uchar green = intensity.val[1];
            uchar red = intensity.val[2];
            unsigned char gray = (77 * red + 150 * green + 29 * blue) >> 8;
            // uchar gray = 0.299 * red + 0.587 * green + 0.114 * blue;
            gray_img.at<uchar>(r, c) = gray;
        }
    }
}

__host__ float compareGrayImages(cv::Mat &bgrImage, cv::Mat &gpuGrayImage) {
    std::cout << "Comparing actual and test grayscale pixel arrays\n";
    cv::Mat cpuGrayImage(bgrImage.rows, bgrImage.cols, CV_8UC1);
    rgb_to_gray_cpu(bgrImage, cpuGrayImage);

    int height = cpuGrayImage.rows;
    int width = cpuGrayImage.cols;

    int numImagePixels = height * width;
    float imagePixelDifference = 0;

    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            imagePixelDifference += abs(cpuGrayImage.at<uchar>(r, c) - gpuGrayImage.at<uchar>(r, c));
        }
    }

    float meanImagePixelDifference = imagePixelDifference / numImagePixels;
    return meanImagePixelDifference;
}

__host__ void cleanUpDevice() {
    std::cout << "Cleaning CUDA device\n";
    CUDA_CHECK(cudaDeviceReset());
}

__global__ void bgr_to_gray_grid1D_block1D(const unsigned char *src, unsigned char *dst, int srcPitch, int dstPitch,
                                           int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total)
        return;

    // Compute 2D coordinates from linear index
    int x = idx % width;
    int y = idx / width;

    // Load the BGR values from pitched memory
    const unsigned char *rowBGR = src + y * srcPitch;
    unsigned char b = rowBGR[x * 3 + 0];
    unsigned char g = rowBGR[x * 3 + 1];
    unsigned char r = rowBGR[x * 3 + 2];

    unsigned char gray = static_cast<unsigned char>(r * 0.299f + g * 0.587f + b * 0.114f);
    // unsigned char gray = (77 * r + 150 * g + 29 * b) >> 8;
    // Store result in pitched destination
    unsigned char *rowG = dst + y * dstPitch;
    rowG[x] = gray;
}

// __global__ void bgr_to_gray_grid1D_block2D(const unsigned char *src, unsigned char *dst, int srcPitch, int dstPitch,
//                                            int width, int height) {}

// __global__ void bgr_to_gray_grid2D_block1D(const unsigned char *src, unsigned char *dst, size_t srcPitch,
//                                            size_t dstPitch, int width, int height) {}

// __global__ void bgr_to_gray_grid2D_block2D(const unsigned char *src, unsigned char *dst, size_t srcPitch,
//                                            size_t dstPitch, int width, int height) {}

__host__ void execute_bgr_to_gray_grid1D_block1D(unsigned char *d_bgr, unsigned char *d_gray, int bgrStepBytes,
                                                 int gryStepBytes, int width, int height) {
    std::cout << "execute bgr_to_gray_grid1D_block1D kernel\n";

    const int threadsPerBlock = 256;

    // Compute total pixels and number of blocks needed
    int totalPixels = width * height;
    int blocks = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;

    bgr_to_gray_grid1D_block1D<<<blocks, threadsPerBlock>>>(d_bgr, d_gray, static_cast<size_t>(bgrStepBytes),
                                                            static_cast<size_t>(gryStepBytes), width, height);

    CUDA_CHECK(cudaGetLastError());

    // Optional: wait for completion
    cudaDeviceSynchronize();
}

// __host__ void execute_bgr_to_gray_grid1D_block2D(unsigned char *d_bgr, unsigned char *d_gray, int bgrStepBytes,
//                                                  int gryStepBytes, int width, int height) {
//     std::cout << "execute bgr_to_gray_grid1D_block2D kernel\n";
// }

// __host__ void execute_bgr_to_gray_grid2D_block1D(uchar *d_r, uchar *d_g, uchar *d_b, uchar *d_gray, int rows,
//                                                  int columns, int threadsPerBlock) {
//     std::cout << "execute bgr_to_gray_grid2D_block1D kernel\n";
// }

// __host__ void execute_bgr_to_gray_grid2D_block2D(uchar *d_r, uchar *d_g, uchar *d_b, uchar *d_gray, int rows,
//                                                  int columns, int threadsPerBlock) {
//     std::cout << "execute bgr_to_gray_grid2D_block2D kernel\n";
// }

int main(int argc, char *argv[]) {
    std::string inputImage = "sloth.png";
    std::string outputImage = "grey-sloth.png";
    cv::Mat bgrImage = cv::imread(inputImage, cv::IMREAD_COLOR);

    size_t width = bgrImage.cols;
    size_t height = bgrImage.rows;

    size_t bgrStepBytes = bgrImage.step[0];
    size_t gryStepBytes = width * sizeof(uchar);

    uchar *d_bgr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bgr, height * bgrStepBytes));
    CUDA_CHECK(cudaMemcpy(d_bgr, bgrImage.data, height * bgrStepBytes, cudaMemcpyHostToDevice));

    uchar *d_gray = nullptr;
    CUDA_CHECK(cudaMalloc(&d_gray, height * gryStepBytes));

    execute_bgr_to_gray_grid1D_block1D(d_bgr, d_gray, bgrStepBytes, gryStepBytes, width, height);
    // execute_bgr_to_gray_grid1D_block2D(d_bgr, d_gray, bgrStepBytes, gryStepBytes, width, height);

    cv::Mat grayImag(height, width, CV_8UC1);
    CUDA_CHECK(cudaMemcpy(grayImag.data, d_gray, height * gryStepBytes, cudaMemcpyDeviceToHost));

    float scaledMeanDifferencePercentage = compareGrayImages(bgrImage, grayImag) * 100;
    std::cout << "Mean difference percentage: " << scaledMeanDifferencePercentage << "\n";

    cv::imwrite(outputImage, grayImag);

    CUDA_CHECK(cudaFree(d_bgr));
    CUDA_CHECK(cudaFree(d_gray));

    cleanUpDevice();
    return 0;
}