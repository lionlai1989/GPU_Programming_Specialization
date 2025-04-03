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

__host__ uchar *cpuConvertToGray(std::string inputImage);
__host__ std::tuple<std::string, std::string, std::string, int> parseCommandLineArguments(int argc, char *argv[]);
__host__ std::tuple<int, int, uchar *, uchar *, uchar *> readImageFromFile(std::string inputFile);
__host__ void copyFromDeviceToHost(uchar *d_gray, uchar *gray, int rows, int columns);
__host__ std::tuple<uchar *, uchar *, uchar *, uchar *> allocateDeviceMemory(int rows, int columns);
__host__ void deallocateMemory(uchar *d_r, uchar *d_g, uchar *d_b, int *d_image_num_pixels);
__host__ void cleanUpDevice();
__host__ void copyFromHostToDevice(uchar *h_r, uchar *d_r, uchar *h_g, uchar *d_g, uchar *h_b, uchar *d_b, int rows, int cols);
__host__ void executeKernel(uchar *d_r, uchar *d_g, uchar *d_b, uchar *d_gray, int rows, int columns, int threadsPerBlock);
__global__ void convert(uchar *d_r, uchar *d_g, uchar *d_b, uchar *d_gray);
__host__ float compareGrayImages(uchar *gray, uchar *test_gray, int rows, int columns);