#include <cassert>        // assert
#include <cuda_runtime.h> // cudaMalloc, cudaMemcpy
#include <iostream>       // cout

#define CUDA_CHECK(err)                                                                                                \
    if ((err) != cudaSuccess) {                                                                                        \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE);                                                                                            \
    }

static constexpr int TILE_SIZE = 16;

__global__ void tiled_multiply_matrix(const uint8_t *__restrict__ A, const float *__restrict__ B, float *__restrict__ C,
                                      int M, int K, int N) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + tx;
    int row = blockIdx.y * TILE_SIZE + ty;

    // Explain why it can't return here
    // if (row >= M || col >= N) {
    //     return;
    // }

    float acc = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        // global indices for this tile
        int aCol = t * TILE_SIZE + tx;
        int bRow = t * TILE_SIZE + ty;

        // load A[row, aCol] if in-bounds, else 0
        if (row < M && aCol < K)
            sA[ty][tx] = float(A[row * K + aCol]);
        else
            sA[ty][tx] = 0.0f;

        // load B[bRow, col] if in-bounds, else 0
        if (bRow < K && col < N)
            sB[ty][tx] = B[bRow * N + col];
        else
            sB[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    // write the result, guard against out-of-bounds
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// rm -rf build && cmake -B build -S . && cmake --build build && ./build/bin/main
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
    dim3 threads_per_block(TILE_SIZE, TILE_SIZE, 1);
    dim3 blocks_per_grid((N + threads_per_block.x - 1) / threads_per_block.x,
                         (M + threads_per_block.y - 1) / threads_per_block.y);
    tiled_multiply_matrix<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, M, K, N);

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
