# GPU Programming Specialization

This repository holds my study notes and hands-on projects for CUDA-based GPU programming. This covers:

- Canny Edge Detector and KLT Tracker with CUDA from scratch
- **GPU Programming Specialization** course by Johns Hopkins University  
- Selected examples from various textbooks  

## [Canny Edge Detector with CUDA from scratch](https://github.com/lionlai1989/GPU_Programming_Specialization/tree/master/Canny_Edge_CUDA)

My CUDA implementation is **3× faster** than OpenCV, even with OpenCV already using 16 threads and SIMD.

![Video Comparison](./Canny_Edge_CUDA/combine_5s.gif)

## [KLT Tracker with CUDA from scratch](https://github.com/lionlai1989/GPU_Programming_Specialization/tree/master/KLT_Tracker_CUDA)

![KLT Tracker naive](./KLT_Tracker_CUDA/tracker_cuda_naive.gif)

## Prerequisites

All code is tested on **Ubuntu 22.04**, using **CUDA 11.8** and **OpenCV 4.7.0**.

### CUDA Toolkit 11.8

Make sure CUDA is installed and environment variables are set:

```bash
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```

### OpenCV 4.7.0 (with FFMPEG + CUDA support)

```bash
# Install FFMPEG and dependencies
sudo apt update && sudo apt install -y libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libgtk2.0-dev libcanberra-gtk-module

# Download opencv and opencv_contrib
wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/4.7.0.zip \
&& wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/4.7.0.zip

# Extract both
unzip opencv.zip && unzip opencv_contrib.zip

cd opencv-4.7.0

# Build and install
rm -rf build/ install_opencv/ \
&& cmake -S . -B build/ \
      -GNinja \
      -DCMAKE_INSTALL_PREFIX=./install_opencv \
      -DCMAKE_BUILD_TYPE=RELEASE \
      -DWITH_CUDA=ON \
      -DWITH_FFMPEG=ON \
      -DWITH_OPENMP=ON \
      -DWITH_OPENCL=ON \
      -DWITH_GTK=ON \
      -DWITH_GTK_2_X=ON \
      -DBUILD_opencv_hdf=OFF \
      -DBUILD_TESTS=OFF \
      -DBUILD_PERF_TESTS=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DOPENCV_GENERATE_PKGCONFIG=ON \
      -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.7.0/modules \
      -DBUILD_LIST=core,cudev,imgproc,imgcodecs,videoio,highgui,video,cudaarithm,cudafilters,cudaimgproc,cudawarping \
&& cmake --build build/ --parallel $(nproc) && cmake --install build/
```

## Resources

- GPU Programming Specialization offered by Johns Hopkins University
