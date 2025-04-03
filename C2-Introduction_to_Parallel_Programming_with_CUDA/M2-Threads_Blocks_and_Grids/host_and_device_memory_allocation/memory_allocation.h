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

__global__ void add(int *d_a, int *d_b, int *d_c, int numElements);
__global__ void sub(int *d_a, int *d_b, int *d_c, int numElements);
__global__ void mult(int *d_a, int *d_b, int *d_c, int numElements);
__global__ void mod(int *d_a, int *d_b, int *d_c, int numElements);
__host__ std::tuple<int *, int *> allocateRandomHostMemory(int numElements);
__host__ std::tuple<int *, int *, int> readCsv(std::string filename);
__host__ std::tuple<int *, int *> allocateDeviceMemory(int numElements);
__host__ void copyFromHostToDevice(int *h_a, int *h_b, int *d_a, int *d_b, int numElements);
__host__ void executeKernel(int *d_a, int *d_b, int *h_c, int numElements, int threadsPerBlock, std::string mathematicalOperation);
__host__ void deallocateMemory(int *d_a, int *d_b);
__host__ void cleanUpDevice();
__host__ void outputToFile(std::string currentPartId, int *h_a, int *h_b, int *h_c, int numElements, std::string mathematicalOperation);
__host__ std::tuple<int, std::string, int, std::string, std::string> parseCommandLineArguments(int argc, char *argv[]);
__host__ std::tuple<int *, int *, int> setUpInput(std::string inputFilename, int numElements);