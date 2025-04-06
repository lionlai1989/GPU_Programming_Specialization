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
#include "cpu_gpu_memory.h"

/*
 * CUDA Kernel Device code
 *
 * Search passed data set for a float value and if the value is at the thread index set the foundIndex value
 */
__global__ void add(int *d_a, int *d_b, int *h_c, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        h_c[i] = d_a[i] + d_b[i];
    }
}

__host__ tuple<int *, int *, int *> allocateRandomHostMemory(int numElements)
{
    size_t size = numElements * sizeof(int);

    // Allocate the host input vector A
    int *h_a = (int *)malloc(size);

    // Allocate the host pinned memory input pointer B
    int *h_b;
    cudaMallocHost((int **)&h_b, size);


    // Allocate the host mapped input vector C
    int *h_c;
    cudaMallocManaged((int **)&h_c, size);

    int maxValue = max((INT_MAX/100), numElements);

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_a[i] = rand() % maxValue;
        h_b[i] = rand() % maxValue;
        h_c[i] = -1;
    }

    return {h_a, h_b, h_c};
}

__host__ tuple<int *, int *> allocateDeviceMemory(int numElements)
{
    size_t size = numElements * sizeof(int);

    // Allocate the device input vector A
    int *d_a = NULL;
    cudaError_t err = cudaMalloc(&d_a, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_a (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Allocate the device input vector B
    int *d_b = NULL;
    err = cudaMalloc(&d_b, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_b (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return {d_a, d_b};
}

__host__ void copyFromHostToDevice(int *h_a, int *h_b, int *d_a, int *d_b, int numElements)
{
    size_t size = numElements * sizeof(int);

    // Paged memory copy
    cudaError_t err = cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Pinned memory copy
    err = cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy constant int d_v from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ void executeKernel(int *d_a, int *d_b, int *h_c, int numElements, int threadsPerBlock)
{
    // Launch the search CUDA Kernel
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, h_c, numElements);
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Free device global memory
__host__ void deallocateMemory(int *d_a, int *d_b)
{

    cudaError_t err = cudaFree(d_a);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_d (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_b);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device int variable d_i (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
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

__host__ tuple<int, int> parseCommandLineArguments(int argc, char *argv[])
{
    // threadsPerBlock numElements
    int numElements = 10;
    int threadsPerBlock = 256;

    for(int i = 1; i < argc; i++)
    {
        std::string option(argv[i]);
        i++;
        std::string value(argv[i]);
        if(option.compare("-n") == 0)
        {
            numElements = atoi(value.c_str());
        }
        else if(option.compare("-t") == 0) 
        {
            threadsPerBlock = atoi(value.c_str());
        }
    }

    return {numElements, threadsPerBlock};
}

/*
 * Host main routine
 */
int main(int argc, char *argv[])
{
    srand(time(0));
    int numElements = 10;
    int threadsPerBlock = 256;

    // cpu_gpu_memory.exe threadsPerBlock numElements
    tuple<int , int> cliArgs = parseCommandLineArguments(argc, argv);
    numElements = get<0>(cliArgs);
    threadsPerBlock = get<1>(cliArgs);

    auto [h_a, h_b, h_c] = allocateRandomHostMemory(numElements);

    printf("Host Input Data: \n");
    for (int i = 0; i < numElements; ++i)
    {
        printf("h_a[%d]: %d h_b[%d]: %d h_c[%d]: %d\n", i, h_a[i], i, h_b[i], i, h_c[i]);
    }

    auto[d_a, d_b] = allocateDeviceMemory(numElements);
    copyFromHostToDevice(h_a, h_b, d_a, d_b, numElements);

    executeKernel(d_a, d_b, h_c, numElements, threadsPerBlock);

    deallocateMemory(d_a, d_b);

    printf("Host Output Data: \n");
    for (int i = 0; i < numElements; ++i)
    {
        printf("h_c[%d]: %d\n", i, h_c[i]);
    }

    cleanUpDevice();
    return 0;
}