#include <stdio.h>
#include <tuple>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <unistd.h>
#include <cuda_runtime.h>

// For the CUDA runtime routines (prefixed with "cuda_")
using namespace std;

__global__ void vectorDiff(const float *a, const float *b, float *c, int numElements);
__host__ std::tuple<float *, float *, float *> allocateHostMemory(int numElements);
__host__ std::tuple<float *, float *> allocateDeviceMemory(int numElements);
__host__ void copyFromHostToDevice(float *hos, float *dev, int numElements);
__host__ void executeKernel(float *d_a, float *d_b, float *c, int numElements);
__host__ void copyFromDeviceToHost(float *dev, float *hos, int numElements);
__host__ void deallocateMemory(float *h_a, float *h_b, float *h_c, float *d_a, float *d_b);
__host__ void cleanUpDevice();
__host__ void placeDataToFiles(float *h_c, int numElements);
__host__ void retrieveDataFromFiles(float *h_a, float *h_b, int numElements);
__host__ void parseFloatsToArrayFromString(float *host_data, std::string line, int numElements);
__host__ std::vector<std::string> split(const std::string &s, char delimiter);
__host__ void performMultiCPUIteration();