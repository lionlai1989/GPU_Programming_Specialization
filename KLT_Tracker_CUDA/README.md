# KLT Tracker with CUDA from scratch

Work in progress.

![Video Comparison](tracker_cuda_naive.gif)

## Source Files

- **tracker_opencv.cpp**  
  Baseline using OpenCV's `cv::calcOpticalFlowPyrLK` on CPU with **16-thread multithreading** and **SIMD**.

- **tracker_basic.cpp**  
  Simple from-scratch C++ version without using `cv::calcOpticalFlowPyrLK` on CPU with single thread and no SIMD. The goal is to make me understand every steps taken in KLT tracker.

- **tracker_cuda_naive.cu**  
  Simple from-scratch CUDA version.

## Performance Evaluation


## Environment Setup

Follow steps in the top-level [`README.md`](https://github.com/lionlai1989/GPU_Programming_Specialization).

## Build and Run

- Build:
```bash
rm -rf build/ && cmake -S . -B build/ && cmake --build build/ -j 8
```

- Run:
```bash
# CPU OpenCV baseline
./build/tracker_opencv

# CPU tutorial baseline. 
./build/tracker_basic

# CUDA variants
```


## Profiling CUDA

Use NVIDIA Nsight Systems to identify kernel and transfer bottlenecks:
