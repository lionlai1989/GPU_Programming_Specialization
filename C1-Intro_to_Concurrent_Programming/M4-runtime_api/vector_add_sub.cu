#include <cmath>   // fabs
#include <cstdlib> // rand
#include <cuda_runtime.h>
#include <iostream> // cout

#define CUDA_CHECK(err)                                                                                                \
    if ((err) != cudaSuccess) {                                                                                        \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE);                                                                                            \
    }

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= numElements) {
        return;
    }
    C[i] = A[i] + B[i];
}

__global__ void vectorSub(const float *A, const float *B, float *C, int numElements) {
    // Unrolled subtraction: each thread handles two elements
    int i = 2 * (blockDim.x * blockIdx.x + threadIdx.x);

    if (i >= numElements) {
        return;
    }

    // Process first element
    C[i] = A[i] - B[i];

    // Process second element if it exists
    if (i + 1 < numElements) {
        C[i + 1] = A[i + 1] - B[i + 1];
    }
}

int main(void) {
    const int numElements = 50000;
    size_t size = numElements * sizeof(float);
    std::cout << "[Vector addition of " << numElements << " elements]" << std::endl;

    std::cout << "Allocate host memory" << std::endl;
    float *h_A = (float *)malloc(numElements * sizeof(float));
    float *h_B = (float *)malloc(numElements * sizeof(float));
    float *h_C = (float *)malloc(numElements * sizeof(float));
    if (h_A == nullptr || h_B == nullptr || h_C == nullptr) {
        std::cerr << "Failed to allocate host vectors!" << std::endl;
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < numElements; ++i) {
        // 0 ~ 1
        h_A[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        h_B[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }

    std::cout << "Allocate device memory" << std::endl;
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, numElements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, numElements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, numElements * sizeof(float)));

    std::cout << "Copy host memory to device memory" << std::endl;
    CUDA_CHECK(cudaMemcpy(d_A, h_A, numElements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, numElements * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "Launch the Vector Add CUDA Kernel" << std::endl;
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "Launch " << blocksPerGrid << " blocks of " << threadsPerBlock << " threads" << std::endl;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    CUDA_CHECK(cudaGetLastError());

    std::cout << "Launch the Vector Sub CUDA Kernel" << std::endl;
    threadsPerBlock = 256;
    blocksPerGrid = (numElements + 2 * threadsPerBlock - 1) /
                    (2 * threadsPerBlock); // Divide by 2 since each thread processes 2 elements
    std::cout << "Launch " << blocksPerGrid << " blocks of " << threadsPerBlock << " threads" << std::endl;
    vectorSub<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_B, d_C, numElements);
    CUDA_CHECK(cudaGetLastError());

    std::cout << "Copy device memory to host memory" << std::endl;
    CUDA_CHECK(cudaMemcpy(h_C, d_C, numElements * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Verify the result: A + B - B = A" << std::endl;
    for (int i = 0; i < numElements; ++i) {
        if (std::fabs(h_C[i] - h_A[i]) > 1e-6) {
            std::cerr << "Result verification failed at element " << i << "!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    std::cout << "Test PASSED" << std::endl;

    // Free device global memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    // Destroy all allocations and reset all state on the current device in the current process.
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
