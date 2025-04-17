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



### M4:

### M5:

### References:
