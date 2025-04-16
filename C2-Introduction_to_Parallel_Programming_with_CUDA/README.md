## Introduction to Parallel Programming with CUDA

### M2: Threads, Blocks and Grids

### M3: Host and Global Memory

#### CUDA Host Memory Model

- **Pageable:** Regular host memory allocated via the OS.
  ```cpp
  int *arr;
  arr = (int *)malloc(1024 * sizeof(int));
  ```

- **Pinned:** Page-locked memory to reduce extra host-device transfers.
  `cudaMallocHost` allocates `size` bytes of host memory that is page-locked and
  accessible to the device. Since the memory can be accessed directly by the
  device, it can be read or written with much higher bandwidth than pageable
  memory obtained with functions such as `malloc`. `arr` can be accessed
  directly by the device and host.
  ```cpp
  float *arr;
  cudaMallocHost((void **)&arr, 1024 * sizeof(float));
  ```

- **Mapped:** Page-locked memory that is directly accessible from the device.
  `cudaHostAlloc` allocates page-locked memory on the host. When flag
  `cudaHostAllocDefault` is used, it's equivalent to calling `cudaMallocHost`.
  When flag `cudaHostAllocMapped` is used, the device pointer can be obtained
  via `cudaHostGetDevicePointer`.
  ```cpp
  float *arr;
  cudaHostAlloc((void **)&arr, 1024 * sizeof(float), cudaHostAllocMapped);
  ```

- **Unified:** Managed memory with automatic data migration between host and
  device. `cudaMallocManaged` allocates memory that will be automatically
  managed by the Unified Memory system. 
  ```cpp
  float *arr;
  cudaMallocManaged((float **)&arr, 1024 * sizeof(float));
  ```

#### CUDA Device Memory Model

- **Allocation:** `cudaMalloc` allocates memory on the device. 
  ```
  cudaMalloc(void **devPtr, size_t size);
  ```

### M4: Shared and Constant Memory

#### shared memory
shared memory in done inside the kernel, since threads in a block share L1
cache.

- **Usage:** Shared among threads in a block (uses L1 cache). Shared memory is
  faster than global memory. Use thread barriers `__syncthreads()` to
  synchronize.
  
- Size known at compile time:
  ```
  __shared__ int arr[10];
  ```
- Size defined at runtime:
  ```
  extern __shared__ int arr[];
  ```

#### Constant Memory

- Characteristics: Read-only memory, globally accessible by all threads. Often
  used in kernels such as those implementing Gaussian blur.
- Declaration:
  ```
  __constant__ int arr[10];
  ```
- Copying data to constant memory:
  ```
  __host__ cudaError_t cudaMemcpyToSymbol(
      const void* symbol, 
      const void* src, 
      size_t count, 
      size_t offset = 0, 
      cudaMemcpyKind kind = cudaMemcpyHostToDevice
  );
  ```

### M5: Register Memory
All variables in kernel functions are allocated to registers by default.
Registers are thread-safe and the fastest memory type. When memory is allocated
beyond register memory, CUDA will need to read/write data from/to cache memory.

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
