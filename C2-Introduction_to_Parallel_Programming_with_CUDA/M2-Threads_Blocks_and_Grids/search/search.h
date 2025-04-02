#include <stdio.h>
#include <tuple>
#include <bits/stdc++.h>
#include <string>
#include <fstream>
#include <vector>
#include <utility>   // std::pair
#include <stdexcept> // std::runtime_error
#include <sstream>   // std::stringstream
#include <ctime>
using namespace std;

// For the CUDA runtime routines (prefixed with "cuda_")

__constant__ int d_v;

__global__ void search(int *d_d, int *d_i, int numElements);
__host__ int *allocateRandomHostMemory(int numElements);
__host__ std::tuple<int *, int *> allocateDeviceMemory(int numElements);
__host__ void copyFromHostToDevice(int h_v, int *h_d, int h_i, int *d_d, int *d_i, int numElements);
__host__ std::tuple<int *, int> readCsv(std::string filename);
__host__ void executeKernel(int *d_d, int *d_i, int numElements, int threadsPerBlock);
__host__ void copyFromDeviceToHost(int *d_i, int &h_i);
__host__ void deallocateMemory(int *h_d, int *d_d, int *d_i);
__host__ void cleanUpDevice();
__host__ void outputToFile(std::string currentPartId, int *data, int numElements, int searchValue, int foundIndex);
__host__ std::tuple<int, int, std::string, int, std::string, bool> parseCommandLineArguments(int argc, char *argv[]);
__host__ std::tuple<int *, int, int> setUpSearchInput(std::string inputFilename, int numElements, int h_v, bool sortInputData);