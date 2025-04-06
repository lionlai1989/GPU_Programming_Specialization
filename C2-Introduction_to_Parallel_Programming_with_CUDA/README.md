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

### M5: Register Memory


### References:

- [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/)

- [CMU GPU Architecture & CUDA Programming](https://www.cs.cmu.edu/afs/cs/academic/class/15418-s18/www/lectures/06_gpuarch.pdf)

- [CUDA Thread Basics by Wake Forest University](https://users.wfu.edu/choss/CUDA/docs/Lecture%205.pdf)

- []()