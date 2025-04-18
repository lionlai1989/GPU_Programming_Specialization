## CUDA Advanced Libraries

### M2: CUDA Fast Fourier Transform (cuFFT)

FFT is O(nlog(n)) and other methods are O(n2). this makes it possible to make
continuous real-time signal processing.

cufft plan creation: simple 1d 2d and 3d.  cufftplan1d 2d 3d

### M3: CUDA Linear Algebra

#### cuBlAS
  - level 1 functions: vector operation
  - level 2 functions: matrix-vector multiplication
  - level 3 functions: matrix-matrix multiplication

#### NVBLAS
The User guide for NVBLAS, drop-in BLAS replacement, multi-GPUs accelerated.
it's built on top of the cuBlAS library.

#### cuSOLVER
  - cuSOLVER - single GPU
  - cuSOLVER - multiple GPUs
  - cuSolverDN, dense linear system
  - cuSolverSP, sparse linear system
  - cuSolverRF, fast re-factorization

#### cuSPARSE: CUDA sparse matrix library
  - all level 1 functions are deprecated. which will be remove by cuda
  - level 2: sparse matrices and dense vectors
  - level 3: sparse matrices and dense matrices

- 

### M4: CUDA Thrust

Thrust is a powerful library of parallel algorithms and data structures.

What is Thrust?

    Thrust is a Nvidia-developed library designed to be similar to C++'s STL and Boost libraries

    Abstracts away some of the lower-level concerns of developing with CUDA

    Main data type is the Vector, there is a host_vector and device_vector

    Uses iterators to allow for simpler parallelization and passing sections of vectors using the begin() and end() methods.

    Less of a need to deal with pointers

Vector Functions

    vector.fill(start, end, value) – fill in the interval with the passed value – use vector.begin() and vector.end() to set all indices

    vector.sequence(start, end, start_val, end_val) – fill in the interval with the sequence of integers between start and end values; if the final two args not included, values are 0, 1, ..., n-1

    vector.copy(start, end, other_vec.start) – copies the values in vector interval [start, end] into other_vec starting at index start

    
### M5: cuDNN and cuTENSOR



### References:
