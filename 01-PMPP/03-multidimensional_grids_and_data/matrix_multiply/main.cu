#include <cassert>        // assert
#include <cuda_runtime.h> // cudaMalloc, cudaMemcpy
#include <iostream>       // cout

#define CUDA_CHECK(err)                                                                                                \
    if ((err) != cudaSuccess) {                                                                                        \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE);                                                                                            \
    }

__global__ void multiply_matrix(uint8_t *a, float *b, float *c, int M, int K, int N) {
    // a: (M, K)
    // b: (K, N)
    // c: (M, N)
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (col >= N || row >= M) {
        return;
    }

    float pix_val = 0.0f;

    for (int i = 0; i < K; ++i) {
        pix_val += a[row * K + i] * b[i * N + col];
    }
    c[row * N + col] = pix_val;
}

// rm -rf build && cmake -B build -S . && cmake --build build && ./build/bin/matrix_multiply
int main() {
    constexpr int M = 200, K = 100, N = 50;
    uint8_t h_a[M * K]; // (M, K)
    float h_b[K * N];   // (K, N)
    float h_c[M * N];   // (M, N)
    for (int i = 0; i < M * K; ++i) {
        h_a[i] = 1;
    }
    for (int i = 0; i < K * N; ++i) {
        h_b[i] = 0.5;
    }
    for (int i = 0; i < M * N; ++i) {
        h_c[i] = 0;
    }

    uint8_t *d_a = nullptr;
    float *d_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, M * K * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_b, K * N * sizeof(float)));

    // H2D
    CUDA_CHECK(cudaMemcpy(d_a, h_a, M * K * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, K * N * sizeof(float), cudaMemcpyHostToDevice));

    float *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_c, M * N * sizeof(float)));

    // Matrix multiplication
    dim3 threads_per_block(16, 16, 1);
    dim3 blocks_per_grid((N + threads_per_block.x - 1) / threads_per_block.x,
                         (M + threads_per_block.y - 1) / threads_per_block.y);
    multiply_matrix<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, M, K, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // D2H
    CUDA_CHECK(cudaMemcpy(h_c, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            // std::cout << h_c[i * N + j] << " ";
            assert(h_c[i * N + j] == 0.5f * K);
        }
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}
