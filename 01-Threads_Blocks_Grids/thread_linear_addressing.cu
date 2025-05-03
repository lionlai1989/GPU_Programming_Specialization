/**
 * Programming in Parallel with CUDA
 * Example 2.1
 */

#include <cstdio>  // printf
#include <cstdlib> // atoi
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

__global__ void vecAdd(int *a, size_t num_elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("vecAdd start tid: %d\n", tid);
    while (tid < num_elements) {
        printf("    vecAdd tid: %d\n", tid);
        a[tid] += 2;
        tid += blockDim.x * gridDim.x; // thread linear addressing
    }
}

__global__ void vecSub(int *a, size_t num_elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("vecSub start tid: %d\n", tid);
    if (tid < num_elements) {
        printf("    vecSub tid: %d\n", tid);
        a[tid] -= 1;
    }
}

// nvcc thread_linear_addressing.cu && ./a.out 1024 32 16 > out_1024_32_16.txt
// nvcc thread_linear_addressing.cu && ./a.out 1030 32 16 > out_1030_32_16.txt
int main(int argc, char *argv[]) {
    int num_elements = std::atoi(argv[1]);
    int threads_per_block = std::atoi(argv[2]);
    int blocks_per_grid = std::atoi(argv[3]);

    int device_id = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max grid dimensions: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", "
              << prop.maxGridSize[2] << ")" << std::endl;

    thrust::device_vector<int> d_sums(num_elements);
    int *d_sums_ptr = thrust::raw_pointer_cast(&d_sums[0]);

    std::cout << "Launch " << blocks_per_grid * threads_per_block << " vecAdd threads. " << "(" << blocks_per_grid
              << "x" << threads_per_block << ")," << "(blocks_per_grid, threads_per_block)" << std::endl;
    // Host is not blocked, so it will not wait for the kernel to finish
    vecAdd<<<blocks_per_grid, threads_per_block>>>(d_sums_ptr, num_elements);

    // Automatically calculate the number of blocks per grid
    blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    std::cout << "Launch " << blocks_per_grid * threads_per_block << " vecSub threads. " << "(" << blocks_per_grid
              << "x" << threads_per_block << ")," << "(blocks_per_grid, threads_per_block)" << std::endl;

    vecSub<<<blocks_per_grid, threads_per_block>>>(d_sums_ptr, num_elements);

    cudaDeviceSynchronize(); // blocks the host until the kernel is finished

    // Uncomment to make assert fail
    // d_sums[num_elements - 1] = 0;

    for (int i = 0; i < num_elements; i++) {
        assert(d_sums[i] == 1);
    }

    return 0;
}
