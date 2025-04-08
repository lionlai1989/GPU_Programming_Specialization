## Introduction to Parallel Programming with CUDA

### M2: Threads, Blocks and Grids

### M3: Host and Global Memory

1. cuda host memory model
- Pageable
    ```cpp
    int size = 1024;
    int *arr;
    arr = (int *)malloc(size * sizeof(int));
    ```

- pinned: removes extra host-side memory transfers
```cpp
int size = 1024;
float *arr;
cudaMallocHost((float **) &arr, size * sizeof(float));
```


- mapped: no copies to device memory
```cpp
cudaHostMalloc((float **) &arr, size * sizeof(float), cudaHostAllocMapped);
```

- unified: no need to worry about copies let system do it for you
```cpp
cudaMallocManaged((int **) &arr, size * sizeof(float));
```

2. cuda device memory
```
cudaMalloc(void ** devPtr, size_t size)
```

```
cudaMemcopy()

cudaHostGetDevicePointer()
```

### M4: Shared and Constant Memory

#### shared memory
shared memory in done inside the kernel, since threads in a block share L1
cache.

if the size is known at compile time:
```
__shared__ int arr[10];
```

if the size is not known at compile time:
```
extern __shared__ int arr[];
```

shared memory is faster than global memory.

synchronize all threads such that all threads stop at thread barriers.
```
__synchthreads();
```

#### constant memory

constant memory is read-only. it's globally accessible on all threads
simultaneously.

```
__constant__ int arr[10];

 __host__ ​cudaError_t cudaMemcpyToSymbol ( const void* symbol, const void* src, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyHostToDevice ) 
```

an example use of constant memory, the kernel of gaussian blur.

### M5: Register Memory


### References:

- [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/)

- [CMU GPU Architecture & CUDA Programming](https://www.cs.cmu.edu/afs/cs/academic/class/15418-s18/www/lectures/06_gpuarch.pdf)

- [CUDA Thread Basics by Wake Forest University](https://users.wfu.edu/choss/CUDA/docs/Lecture%205.pdf)

- []()