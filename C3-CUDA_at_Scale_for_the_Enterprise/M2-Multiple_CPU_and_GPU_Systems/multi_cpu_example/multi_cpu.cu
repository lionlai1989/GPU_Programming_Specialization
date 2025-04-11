/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * Vector multiplication: C = A * B.
 *
 * This sample is a very basic sample that implements element by element
 * vector multiplication. It is based on the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include "multi_cpu.h"

/*
 * CUDA Kernel Device code
 *
 * Computes the differences between values at all indexes of a and b into c. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void vectorDiff(const float *a, const float *b, float *c, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        c[i] = a[i] - b[i];
    }
}

__host__ std::tuple<float *, float *, float *> allocateHostMemory(int numElements)
{
    size_t size = numElements * sizeof(float);

    // Allocate the host input vector a
    float *h_a = (float *)malloc(size);

    // Allocate the host input vector b
    float *h_b = (float *)malloc(size);

    // Allocate the host output vector c
    float *h_c;
    cudaMallocManaged((float **)&h_c, size);


    // Verify that allocations succeeded
    if (h_a == NULL || h_b == NULL || h_c == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    return {h_a, h_b, h_c};
}

__host__ std::tuple<float *, float *> allocateDeviceMemory(int numElements)
{
    // Allocate the device input vector a
    float *d_a = NULL;
    size_t size = numElements * sizeof(float);
    cudaError_t err = cudaMalloc(&d_a, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector a (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector b
    float *d_b = NULL;
    err = cudaMalloc(&d_b, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector b (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return {d_a, d_b};
}

__host__ void copyFromHostToDevice(float *hos, float *dev, int numElements)
{
    size_t size = numElements * sizeof(float);
    // Copy the host input vector to the device input vectors
    printf("Copy input data from the host memory to the CUDA device\n");
    cudaError_t err = cudaMemcpy(dev, hos, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector data from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ void executeKernel(float *d_a, float *d_b, float *c, int numElements)
{
    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    // REPLACE x, y, z with a, b, and c variables for memory on the GPU
    vectorDiff<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, c, numElements);
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


// Free device global memory
__host__ void deallocateMemory(float *h_a, float *h_b, float *h_c, float *d_a, float *d_b)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaFree(d_a);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector a (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_b);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector b (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);
}

// Reset the device and exit
__host__ void cleanUpDevice()
{
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaError_t err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Based on http://www.cplusplus.com/forum/general/170845/
__host__ void placeDataToFiles(float *h_c, int numElements)
{
    printf("Placing calculation results into output files\n");
    ofstream outfile_a;
	outfile_a.open("./output_a.csv");
    ofstream outfile_b;
	outfile_b.open("./output_b.csv");

    // print first column's element
    outfile_a << h_c[0];
    outfile_b << -(h_c[0]);

    for (int i=1; i < numElements; i++)
    {
        outfile_a << "," << h_c[i];
        outfile_b << "," << (0-h_c[i]);
    }

    outfile_a << endl;
    outfile_b << endl;
}

// Based on content found at https://www.cplusplus.com/reference/cstdio
__host__ void retrieveDataFromFiles(float *h_a, float *h_b, int numElements)
{
    printf("Retrieving data from input files.\n");

    // Wait for lock files to exist, so know that data is in expected files and then remove output file and lock
    bool locksExist = false;

    while(!locksExist)
    {
        std::ifstream lock_a("./input_a.lock");
        std::ifstream lock_b("./input_b.lock");
        if(lock_a.is_open() && lock_b.is_open())
        {
            locksExist = true;
        }
        sleep(10);
    }
    
    printf("Removing output files.\n");
    remove( "./output_a.csv" );
    remove( "./output_a.lock" );
    remove( "./output_b.csv" );
    remove( "./output_b.lock" );

    string line_a;
    string line_b;

    printf("Parsing array from input csv files.\n");
    ifstream file_a ("./input_a.csv");
    ifstream file_b ("./input_b.csv");
    if (file_a.is_open() && file_b.is_open())
    {
        getline (file_a,line_a);
        printf("Parsing line: %s\n",line_a.c_str());
        parseFloatsToArrayFromString(h_a, line_a, numElements);
        getline (file_b,line_b);
        printf("Parsing line: %s\n",line_b.c_str());
        parseFloatsToArrayFromString(h_b, line_b, numElements);
    }
}

__host__ void parseFloatsToArrayFromString(float * host_data, std::string line, int numElements)
{
    printf ("Splitting string \"%s\" into tokens\n",line.c_str());
    std::istringstream iss(line);
    std::string token;
    int i = 0;
    while (std::getline(iss, token, ','))
    {
        printf ("%s,",token.c_str());
        host_data[i] = std::stof(token);
        i++;
    }
    printf ("\n");
}

__host__ std::vector<std::string> split(const std::string& s, char delimiter)
{
   std::vector<std::string> tokens;
   std::string token;
   std::istringstream tokenStream(s);
   while (std::getline(tokenStream, token, delimiter))
   {
      tokens.push_back(token);
   }
   return tokens;
}

__host__ void performMultiCPUIteration()
{
    int numElements = 128;
    printf("Vector difference of %d elements\n", numElements);

    auto[h_a, h_b, h_c] = allocateHostMemory(numElements);
    auto[d_a, d_b] = allocateDeviceMemory(numElements);

    retrieveDataFromFiles(h_a, h_b, numElements);

    copyFromHostToDevice(h_a, d_a, numElements);
    copyFromHostToDevice(h_b, d_b, numElements);

    executeKernel(d_a, d_b, h_c, numElements);
    printf("Completed execution kernel\n");

    placeDataToFiles(h_c, numElements);
    cudaDeviceSynchronize();

    // Remove locks on input and lock on output, which should signal consumers to read
    remove( "./input_a.lock" );
    remove( "./input_b.lock" );

    fstream output_fstream_a;
    output_fstream_a.open("output_a.lock", std::ios_base::out);
    output_fstream_a.is_open();
    fstream output_fstream_b;
    output_fstream_b.open("output_b.lock", std::ios_base::out);
    output_fstream_b.is_open();

}

/*
 * Host main routine
 */
int main(void)
{
    while(true)
    {
        performMultiCPUIteration();
    }
    cleanUpDevice();
    printf("Done\n");
    return 0;
}