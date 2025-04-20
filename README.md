# GPU Programming Specialization

This repo contains my study notes for the **GPU Programming Specialization** offered by Johns Hopkins University on Coursera.

All code is tested on **Ubuntu 22.04**, with **CUDA 11.8** and **OpenCV 4.7.0**.

## KLT Tracker with CUDA from scratch

I use CUDA libraries build a KLT tracker from scratch.

## Prerequisites

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
rm -rf build/ \
&& cmake -S . -B build/ \
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
      -DBUILD_LIST=core,cudev,imgproc,imgcodecs,videoio,highgui,video \
&& cmake --build build/ --parallel $(nproc) && cmake --install build/
```

## Course Description
// please fill in this part
### Introduction to Concurrent Programming with GPUs
// please fill in this part

### Introduction to Parallel Programming with CUDA
// please fill in this part

### CUDA at Scale for the Enterprise
// please fill in this part

### CUDA Advanced Libraries
// please fill in this part
