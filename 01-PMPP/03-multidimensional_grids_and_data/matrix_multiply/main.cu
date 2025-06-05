#include <cuda_runtime.h>     // cudaMalloc, cudaMemcpy
#include <iostream>           // cout
#include <opencv2/opencv.hpp> // imread, imwrite

#define CUDA_CHECK(err)                                                                                                \
    if ((err) != cudaSuccess) {                                                                                        \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE);                                                                                            \
    }

__global__ void multiply_matrix(uint8_t *mat1, float *mat2, float *out, int width) {
    // mat1 * mat2
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (col >= width || row >= width) {
        return;
    }

    float pix_val = 0.0f;

    // (y, x)
    for (int k = 0; k < width; ++k) {
        pix_val += mat1[row * width + k] * mat2[k * width + col];
    }
    out[row * width + col] = pix_val;
}

// rm -rf build && cmake -B build -S . && cmake --build build && ./build/bin/matrix_multiply
int main() {
    int width = 20;
    uint8_t *d_mat1 = nullptr; // 20x20
    float *d_mat2 = nullptr;   // 20x20
    CUDA_CHECK(cudaMalloc(&d_mat1, width * width * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_mat2, width * width * sizeof(float)));

    float *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output, width * width * sizeof(float)));

    // Convert to grayscale
    dim3 threads_per_block(16, 16, 1);
    dim3 blocks_per_grid((width + threads_per_block.x - 1) / threads_per_block.x,
                         (width + threads_per_block.y - 1) / threads_per_block.y);
    multiply_matrix<<<blocks_per_grid, threads_per_block>>>(d_mat1, d_mat2, d_output, width);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_mat1));
    CUDA_CHECK(cudaFree(d_mat2));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
