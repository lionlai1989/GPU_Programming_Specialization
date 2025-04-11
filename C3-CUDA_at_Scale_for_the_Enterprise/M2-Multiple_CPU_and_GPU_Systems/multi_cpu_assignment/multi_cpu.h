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

__global__ void compare(const int *a, const int *b, int *c, int numElements);
__host__ std::tuple<int *, int *, int *> allocateHostMemory(int numElements);
__host__ std::tuple<int *, int *> allocateDeviceMemory(int numElements);
__host__ void copyFromHostToDevice(int *hos, int *dev, int numElements);
__host__ void executeKernel(int *d_a, int *d_b, int *c, int numElements);
__host__ void copyFromDeviceToHost(int *dev, int *hos, int numElements);
__host__ void deallocateMemory(int *h_a, int *h_b, int *h_c, int *d_a, int *d_b);
__host__ void cleanUpDevice();
__host__ void placeDataToFiles(int *h_c, int numElements);
__host__ void retrieveDataFromFiles(int *h_a, int *h_b, int numElements);
__host__ void parseIntsToArrayFromString(int *host_data, std::string line, int numElements);
__host__ std::vector<std::string> split(const std::string &s, char delimiter);
__host__ void performMultiCPUIteration();