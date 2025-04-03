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

const float LO_RAND = 0.0;
const float HI_RAND = 100.0;

__global__ void div(float *d_a, float *d_b, float *d_c, int numElements);
__host__ std::tuple<float *, float *> allocateRandomHostMemory(int numElements);
__host__ std::tuple<float *, float *, int> readCsv(std::string filename);
__host__ std::tuple<float *, float *> allocateDeviceMemory(int numElements);
__host__ void copyFromHostToDevice(float *h_a, float *h_b, float *d_a, float *d_b, int numElements);
__host__ void executeKernel(float *d_a, float *d_b, float *h_c, int numElements, int threadsPerBlock);
__host__ void deallocateMemory(float *d_a, float *d_b);
__host__ void cleanUpDevice();
__host__ void outputToFile(std::string currentPartId, float *h_a, float *h_b, float *h_c, int numElements);
__host__ std::tuple<int, std::string, int, std::string, std::string> parseCommandLineArguments(int argc, char *argv[]);
__host__ std::tuple<float *, float *, int> setUpInput(std::string inputFilename, int numElements);