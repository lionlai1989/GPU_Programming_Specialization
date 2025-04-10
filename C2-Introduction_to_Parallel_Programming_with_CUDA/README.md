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
all variable are allocated as regster memory in kernel function.
thread-save memory.
when memory is allocated beyond register memory, CUDA will need to read/write
data from/to cache memory.

### Device Memory comparison

Below are short descriptions for each memory type, using NVIDIA RTX A3000
specifications:

- Global Memory: The RTX A3000 features 6 GB of GDDR6 global memory. It offers a
  large capacity for all non-constant data, but access is slower compared to
  on-chip memories.

- Constant Memory: This GPU provides 64 KB of constant memory, which is
  optimized for broadcast access to read-only data shared across all threads.

- Shared Memory: Each SM on the RTX A3000 can be configured with up to 96 KB of
  low-latency shared memory. It enables fast communication and data sharing
  among threads within a block.

- Register Memory: The register file per SM on the RTX A3000 is extremely fast,
  offering around 256 KB for thread-local and thread-save access. Although registers are
  limited per thread, they provide the quickest access for temporary variables.

Streaming Multiprocessor (SM)—a core processing unit within an NVIDIA GPU.
Each SM contains many CUDA cores, registers, shared memory, and other resources,
allowing it to execute many threads in parallel.

### References:

- [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/)

- [CMU GPU Architecture & CUDA Programming](https://www.cs.cmu.edu/afs/cs/academic/class/15418-s18/www/lectures/06_gpuarch.pdf)

- [CUDA Thread Basics by Wake Forest University](https://users.wfu.edu/choss/CUDA/docs/Lecture%205.pdf)

- []()