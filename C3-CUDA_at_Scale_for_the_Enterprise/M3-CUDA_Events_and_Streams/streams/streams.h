#include <stdio.h>
#include <tuple>
#include <string>
#include <cuda_runtime.h>

// For the CUDA runtime routines (prefixed with "cuda_")
using namespace std;

__global__ void kernelA1(float *dev_mem, int n);
__global__ void kernelB1(float *dev_mem, int n);
__global__ void kernelA2(float *dev_mem, int n);
__global__ void kernelB2(float *dev_mem, int n);

__host__ float *allocateHostMemory(int numElements, int seed);
__host__ float *allocateDeviceMemory(int numElements);
__host__ void copyFromHostToDeviceSync(float *host_mem, float *dev_mem, int numElements, cudaStream_t stream);
__host__ void copyFromHostToDeviceAsync(float *host_mem, float *dev_mem, int numElements, cudaStream_t stream);
__host__ void copyFromDeviceToHostSync(float *dev_mem, float *host_mem, int numElements, cudaStream_t stream);
__host__ void copyFromDeviceToHostAsync(float *dev_mem, float *host_mem, int numElements, cudaStream_t stream);
__host__ void deallocateDevMemory(float *dev_mem);
__host__ void cleanUpDevice();
__host__ std::tuple<int, int> determineThreadBlockDimensions(int num_elements);
__host__ float *runStreamsFullAsync(float *host_mem, int num_elements);
__host__ float *runStreamsBlockingKernel2StreamsNaive(float *host_mem, int num_elements);
__host__ float *runStreamsBlockingKernel2StreamsOptimal(float *host_mem, int num_elements);
__host__ void printHostMemory(float *host_mem, int num_elments);
