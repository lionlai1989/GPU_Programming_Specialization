
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <tuple>

#define CUDA_CHECK(err)                                                                                                \
    if ((err) != cudaSuccess) {                                                                                        \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE);                                                                                            \
    }

/**
 * The __device__ execution space specifier declares a function that is executed on the device and can only be called
 * from other device functions.
 */
__device__ float deviceMultiply(float a, float b) { return a * b; }

/**
 * The __global__ execution space specifier declares a function as being a kernel, which is executed on the device and
 * can be called from the host.
 */
__global__ void vectorMult(const float *A, const float *B, float *C, int numElements) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= numElements) {
        return;
    }
    C[tid] = deviceMultiply(A[tid], B[tid]);
}

/**
 * The __host__ execution space specifier declares a function that is executed on the host and can be called from the
 * host.
 */
__host__ std::tuple<float *, float *, float *> allocateHostMemory(int numElements) {
    size_t size = numElements * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    if (h_A == nullptr || h_B == nullptr || h_C == nullptr) {
        std::cerr << "Failed to allocate host vectors!" << std::endl;
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numElements; ++i) {
        // 0 ~ 1. 0 <= rand() <= RAND_MAX
        h_A[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        h_B[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }

    return {h_A, h_B, h_C};
}

__host__ std::tuple<float *, float *, float *> allocateDeviceMemory(int numElements) {
    size_t size = numElements * sizeof(float);

    float *d_A = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    float *d_B = nullptr;
    CUDA_CHECK(cudaMalloc(&d_B, size));
    float *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_C, size));
    return {d_A, d_B, d_C};
}

__host__ void copyFromHostToDevice(float *h_A, float *h_B, float *d_A, float *d_B, int numElements) {
    size_t size = numElements * sizeof(float);

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
}

__host__ void executeKernel(float *d_A, float *d_B, float *d_C, int numElements) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "CUDA kernel launch with " << blocksPerGrid << " blocks of " << threadsPerBlock << " threads"
              << std::endl;

    vectorMult<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    CUDA_CHECK(cudaGetLastError());
}

__host__ void copyFromDeviceToHost(float *d_C, float *h_C, int numElements) {
    size_t size = numElements * sizeof(float);
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
}

__host__ void deallocateDeviceMemory(float *d_A, float *d_B, float *d_C) {
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

__host__ void deallocateHostMemory(float *h_A, float *h_B, float *h_C) {
    free(h_A);
    free(h_B);
    free(h_C);
}

__host__ void performTest(float *h_A, float *h_B, float *h_C, int numElements) {
    for (int i = 0; i < numElements; ++i) {
        if (fabs((h_A[i] * h_B[i]) - h_C[i]) > 1e-6) {
            std::cerr << "Result verification failed at element " << i << "!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    std::cout << "Test PASSED" << std::endl;
}

int main(void) {
    int numElements = 50000;
    std::cout << "[Vector multiplication of " << numElements << " elements]" << std::endl;

    auto [h_A, h_B, h_C] = allocateHostMemory(numElements);
    auto [d_A, d_B, d_C] = allocateDeviceMemory(numElements);

    copyFromHostToDevice(h_A, h_B, d_A, d_B, numElements);
    executeKernel(d_A, d_B, d_C, numElements);
    copyFromDeviceToHost(d_C, h_C, numElements);

    performTest(h_A, h_B, h_C, numElements);

    deallocateHostMemory(h_A, h_B, h_C);
    deallocateDeviceMemory(d_A, d_B, d_C);

    CUDA_CHECK(cudaDeviceReset());
    std::cout << "Done" << std::endl;
    return 0;
}