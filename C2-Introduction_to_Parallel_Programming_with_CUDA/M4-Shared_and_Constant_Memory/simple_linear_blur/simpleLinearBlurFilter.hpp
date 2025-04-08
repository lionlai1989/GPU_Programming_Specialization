#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <stdio.h>
#include <tuple>
#include <string>

using namespace cv;
using namespace std;

__device__ __constant__ int d_rows;
__device__ __constant__ int d_columns;

__global__ void applySimpleLinearBlurFilter(uchar *r, uchar *g, uchar *b);
__host__ float compareColorImages(uchar *r0, uchar *g0, uchar *b0, uchar *r1, uchar *g1, uchar *b1, int rows, int columns);
__host__ void allocateDeviceMemory(int rows, int columns);
__host__ void executeKernel(uchar *r, uchar *g, uchar *b, int rows, int columns, int threadsPerBlock);
__host__ void cleanUpDevice();
__host__ std::tuple<std::string, std::string, std::string, int> parseCommandLineArguments(int argc, char *argv[]);
__host__ std::tuple<int, int, uchar *, uchar *, uchar *> readImageFromFile(std::string inputFile);
__host__ std::tuple<uchar *, uchar *, uchar *> applyBlurKernel(std::string inputImage);
int main(int argc, char *argv[]);