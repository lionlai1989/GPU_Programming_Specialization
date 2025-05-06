# Canny Edge CUDA

![Video Comparison](combine.gif)

## Source Files

- **canny_opencv.cpp**  
  Baseline using OpenCV's `cv::Canny` on CPU with **16-thread multithreading** and **SIMD**.

- **canny_cuda_naive.cu**  
  Simple from-scratch CUDA version with direct global-memory accesses.

- **canny_cuda_encapsulate.cu**  
  Encapsulate global buffers into a class to reuse device memory and reduce allocation overhead.

- **canny_cuda_pinned.cu**  
  Builds on the encapsulated version, adding page-locked ("pinned") host buffers to speed host–device transfers.

The fundamental framework of all source files do the following three things:

- A reader thread pulls raw frames from the input video and feeds them into a thread-safe reader queue.

- The main thread fetches frames from the reader queue, transfers data to the GPU, launches and synchronizes the Canny kernels, then pushes the resulting edge maps into the writer queue.

- A writer thread retrieves processed frames from the writer queue and writes them sequentially to the output video.

## Performance Evaluation

Performance tests are conducted using two 360° video samples:
- High-resolution: 3840×1920 pixels, 30 FPS, 100 seconds duration
- Low-resolution: 1920×960 pixels, 30 FPS, 100 seconds duration

Performance metrics were measured for each implementation:

| Implementation | High Resolution<br>(3840×1920) |  | Low Resolution<br>(1920×960) |  |
| -------------- | :-----------------------------: | :-: | :--------------------------: | :-: |
|                | Per-Frame (μs)                 | Total (ms) | Per-Frame (μs)             | Total (ms) |
| **OpenCV**     | –                              | –     | –                          | –     |
| **Naive**      | –                              | –     | –                          | –     |
| **Encapsulate**| –                              | –     | –                          | –     |
| **Pinned**     | –                              | –     | –                          | –     |

*All values were measured over 10 runs, and the average of the best three results is reported.*

## Environment Setup

Follow steps in the top-level [`README.md`](https://github.com/lionlai1989/GPU_Programming_Specialization).

## Build and Run

- Build:
```bash
rm -rf build/ && cmake -S . -B build/ && cmake --build build/ -j 8
```

- Run:
```bash
# CPU baseline
./build/canny_opencv

# CUDA variants
./build/canny_cuda_naive
./build/canny_cuda_encasulate
./build/canny_cuda_pinned
```

- Sample Output
```text
Processing video: 1920x960 @ 30 fps with 3000 frames
Start timing ...
Total time of all frame: 35774854 microseconds. Each frame average time: 11924 microseconds.
End-to-end runtime: 85040 milliseconds
Done. Output saved as canny_opencv.mp4
```

  - `11924 microseconds`: GPU work divided by frame count,
  - `35774854 microseconds`: the sum of all kernel execution durations across frames.
  - `85040 milliseconds`: end-to-end runtime, includes video I/O plus GPU processing, reflecting real-world throughput.

## Profiling CUDA

Use NVIDIA Nsight Systems to identify kernel and transfer bottlenecks:

```bash
nsys profile \
--trace=cuda,nvtx,osrt \
--trace-fork-before-exec=true \
--output=canny_cuda_naive \
./build/canny_cuda_naive

nsys stats canny_cuda_naive.nsys-rep
```

### Compare Results
You can concatenate the input and output videos horizontally for comparison using FFMPEG:

```bash
# Horizontal stack: -filter_complex "[0:v][1:v]hstack=inputs=2[v]"
# Vertical stack:   -filter_complex "[0:v][1:v]vstack=inputs=2[v]"
ffmpeg \
  -i data/me.mp4 \
  -i output/canny_edge.mp4 \
  -filter_complex "[0:v][1:v]vstack=inputs=2[v]" \
  -map "[v]" \
  -vsync 2 \
  -c:v libx264 \
  -crf 23 \
  -preset fast \
  -x264-params level=6.2 \
  combine.mp4
```

The comparison is shown as a gif below:

```bash
# First, create a GIF
ffmpeg -i combine.mp4 -vf "fps=15,scale=640:-1:flags=lanczos" -loop 0 combine.gif
```
