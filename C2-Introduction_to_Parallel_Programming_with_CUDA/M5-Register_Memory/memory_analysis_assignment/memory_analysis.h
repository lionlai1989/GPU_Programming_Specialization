#include <iostream>
#include <stdio.h>
#include <tuple>
#include <string>
#include <math.h>
#include <fstream>
#include <stdexcept> // std::runtime_error
#include <sstream>   // std::stringstream

using namespace std;

__device__ __constant__ int *constant_input;
__device__ __constant__ int constant_search_value;
__device__ __constant__ int constant_num_elements;
__device__ __constant__ int constant_num_threads;
__device__ __constant__ int constant_thread_span;

const int NUM_KERNELS = 1;
const std::string KERNEL_TYPES[NUM_KERNELS] = {"global"};

__global__ void globalMemorySearch(int *input, int totalFound, int numElements, int numThreads);
__global__ void constantMemorySearch(int totalFound);
__global__ void sharedMemorySearch(int *input, int totalFound, int numElements, int numThreads);
__global__ void registerMemorySearch(int *input, int totalFound, int numElements, int numThreads);

__host__ int *allocatePageableRandomHostMemory(int numElements);
__host__ int *allocateDeviceMemory(int numElements);
__host__ void copyFromHostToDevice(std::string kernelType, int *input, int *numElements, int numThreads, int *d_input);
__host__ void executeKernel(int *d_input, int numElements, int numThreads, int threadsPerBlock, std::string kernelType);
__host__ void deallocateMemory(int *d_input);
__host__ void cleanUpDevice();
__host__ std::tuple<int, std::string, int, std::string> parseCommandLineArguments(int argc, char *argv[]);
__host__ int *setUpInput(int numElements);