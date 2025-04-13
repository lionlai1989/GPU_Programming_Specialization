// Based on code found at https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf
#include "streams.h"

// Increments all of the values in the input arrays
__global__ void kernelA1(float *dev_mem, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x)
    {
        dev_mem[i] = dev_mem[i] + 1;
    }
}

// Doubles all the values in the input arrays
__global__ void kernelB1(float *dev_mem, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x)
    {
        dev_mem[i] = dev_mem[i] * 2;
    }
}

// Decrements all of the values in the input arrays
__global__ void kernelA2(float *dev_mem, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x)
    {
        dev_mem[i] = dev_mem[i] - 1;
    }
}

// Halves all the values in the input arrays
__global__ void kernelB2(float *dev_mem, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x)
    {
        dev_mem[i] = dev_mem[i] / 2;
    }
}

// This will generate an array of size numElements of random integers from 0 to 255 in pageable host memory
// The host memory has to be page-locked memory or control of streams is not guaranteed
// Note that I have added an argument for the random seed, so that you can generate the same "random" values
// for multiple runs to see the result of different actions on the same set of "random" values
__host__ float *allocateHostMemory(int numElements, int seed)
{
    seed = seed != -1 ? seed : 0;
    srand(seed);
    size_t size = numElements * sizeof(float);
    float random_max = 255.0f;

    // Allocate the host pinned memory input pointer B
    float *data;
    cudaHostAlloc((void **)&data, size, cudaHostAllocDefault);

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        // Feel free to change the max value of the random input data by replacing 255 with a smaller or larger number
        data[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / random_max));
    }

    return data;
}

__host__ float *allocateDeviceMemory(int numElements)
{
    // Allocate the device input vector a
    float *dev_mem = NULL;
    size_t size = numElements * sizeof(float);
    cudaError_t err = cudaMalloc(&dev_mem, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector memory (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return dev_mem;
}

// Synchronous copy of data from host to device using a default stream
__host__ void copyFromHostToDeviceSync(float *host_mem, float *dev_mem, int numElements)
{
    size_t size = numElements * sizeof(float);
    // Copy the host input vector to the device input vectors
    printf("Copy input data from the host memory to the CUDA device\n");
    cudaError_t err = cudaMemcpy(dev_mem, host_mem, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector data from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Asynchronous copy of data from host to device using a non-default stream
__host__ void copyFromHostToDeviceAsync(float *host_mem, float *dev_mem, int numElements, cudaStream_t stream)
{
    size_t size = numElements * sizeof(float);
    // Copy the host input vector to the device input vectors
    printf("Copy input data from the host memory to the CUDA device\n");
    cudaError_t err = cudaMemcpyAsync(dev_mem, host_mem, size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector data from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Synchronous copy of data from device to host using the default stream
__host__ void copyFromDeviceToHostSync(float *dev_mem, float *host_mem, int numElements)
{
    size_t size = numElements * sizeof(float);
    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    cudaError_t err = cudaMemcpy(host_mem, dev_mem, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Synchronous copy of data from device to host using a non-default stream
__host__ void copyFromDeviceToHostAsync(float *dev_mem, float *host_mem, int numElements, cudaStream_t stream)
{
    size_t size = numElements * sizeof(float);
    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    cudaError_t err = cudaMemcpyAsync(host_mem, dev_mem, size, cudaMemcpyDeviceToHost, stream);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Free device global memory
__host__ void deallocateDevMemory(float *dev_mem)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaFree(dev_mem);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector (error code %s)!\n", cudaGetErrorString(err));
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

__host__ std::tuple<int, int> determineThreadBlockDimensions(int num_elements)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    return {threadsPerBlock, blocksPerGrid};
}

__host__ float *runStreamsFullAsync(float *host_mem, int num_elements)
{
    // Prepare all streams such that all kernels and memory copies execute asynchronously
    cudaStream_t stream1, stream2, stream3, stream4, stream5, stream6;
    cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream3, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream4, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream5, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream6, cudaStreamNonBlocking);

    // Prepare device memory based on host memory
    float *dev_mem = allocateDeviceMemory(num_elements);
    copyFromHostToDeviceAsync(host_mem, dev_mem, num_elements, stream1);

    // Execute 4 kernels asynchronously on independent streams
    auto [threadsPerBlock, blocksPerGrid] = determineThreadBlockDimensions(num_elements);
    kernelA1<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(dev_mem, num_elements);
    kernelB1<<<blocksPerGrid, threadsPerBlock, 0, stream3>>>(dev_mem, num_elements);
    kernelA2<<<blocksPerGrid, threadsPerBlock, 0, stream4>>>(dev_mem, num_elements);
    kernelB2<<<blocksPerGrid, threadsPerBlock, 0, stream5>>>(dev_mem, num_elements);

    // Copy device memory back to host asynchronously
    copyFromDeviceToHostAsync(dev_mem, host_mem, num_elements, stream6);
    deallocateDevMemory(dev_mem);

    // Wait for all streams to be completed
    // This might act differently on multiple GPU system without assigning devices
    cudaDeviceSynchronize();

    return host_mem;
}

__host__ float *runStreamsBlockingKernel2StreamsNaive(float *host_mem, int num_elements)
{
    // Prepare all streams such that all kernels and memory copies execute asynchronously
    cudaStream_t stream1, stream2, stream3, stream4;
    cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream3, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream4, cudaStreamNonBlocking);

    // Prepare device memory based on host memory
    float *dev_mem = allocateDeviceMemory(num_elements);
    copyFromHostToDeviceAsync(host_mem, dev_mem, num_elements, stream1);

    // Execute 2 pairs of kernels asynchronous with resepect to their streams
    // The order of execution can have an effect on the blocking behavious
    auto [threadsPerBlock, blocksPerGrid] = determineThreadBlockDimensions(num_elements);
    kernelA1<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(dev_mem, num_elements);
    kernelB1<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(dev_mem, num_elements);
    kernelA2<<<blocksPerGrid, threadsPerBlock, 0, stream3>>>(dev_mem, num_elements);
    kernelB2<<<blocksPerGrid, threadsPerBlock, 0, stream3>>>(dev_mem, num_elements);

    // Copy device memory back to host asynchronously
    copyFromDeviceToHostAsync(dev_mem, host_mem, num_elements, stream4);
    deallocateDevMemory(dev_mem);

    // Wait for all streams to be completed
    // This might act differently on multiple GPU system without assigning devices
    cudaDeviceSynchronize();

    return host_mem;
}

__host__ float *runStreamsBlockingKernel2StreamsOptimal(float *host_mem, int num_elements)
{
    // Prepare all streams such that all kernels and memory copies execute asynchronously
    cudaStream_t stream1, stream2, stream3, stream4;
    cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream3, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream4, cudaStreamNonBlocking);

    // Prepare device memory based on host memory
    float *dev_mem = allocateDeviceMemory(num_elements);
    copyFromHostToDeviceAsync(host_mem, dev_mem, num_elements, stream1);

    // Execute 2 pairs of kernels asynchronous with resepect to their streams
    // The order of execution can have an effect on the blocking behavious
    auto [threadsPerBlock, blocksPerGrid] = determineThreadBlockDimensions(num_elements);
    kernelA1<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(dev_mem, num_elements);
    kernelA2<<<blocksPerGrid, threadsPerBlock, 0, stream3>>>(dev_mem, num_elements);
    kernelB1<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(dev_mem, num_elements);
    kernelB2<<<blocksPerGrid, threadsPerBlock, 0, stream3>>>(dev_mem, num_elements);

    // Copy device memory back to host asynchronously
    copyFromDeviceToHostAsync(dev_mem, host_mem, num_elements, stream4);
    deallocateDevMemory(dev_mem);

    // Wait for all streams to be completed
    // This might act differently on multiple GPU system without assigning devices
    cudaDeviceSynchronize();

    return host_mem;
}

__host__ void printHostMemory(float *host_mem, int num_elments)
{
    // Output results
    for (int i = 0; i < num_elments; i++)
    {
        printf("%.6f ", host_mem[i]);
    }
    printf("\n");
}

int main()
{
    int num_elements = 255; // Can be altered but keep it less than 1/2 the memory size of global memory for full concurrency
    int rand_seed = 0;      // You can set this to different values for each run but default will be the same to see the effect on data

    float *host_mem = allocateHostMemory(num_elements, rand_seed);
    host_mem = runStreamsFullAsync(host_mem, num_elements);
    printHostMemory(host_mem, num_elements);

    host_mem = allocateHostMemory(num_elements, 0);
    host_mem = runStreamsBlockingKernel2StreamsNaive(host_mem, num_elements);
    printHostMemory(host_mem, num_elements);

    host_mem = allocateHostMemory(num_elements, 0);
    host_mem = runStreamsBlockingKernel2StreamsOptimal(host_mem, num_elements);
    printHostMemory(host_mem, num_elements);

    return 0;
}