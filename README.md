# GPU_Programming_Specialization

this repository has my study notes for the course, GPU Programming Specialization, offered by johns hopkins university and coursera.

All the code can be compiled and run if you have already set up the environment. all code has beend tested on ubuntu22.04 cuda 11.8 and opencv4.7.0. 

## Description
// please fill in this part
### Introduction to Concurrent Programming with GPUs
// please fill in this part

### Introduction to Parallel Programming with CUDA
// please fill in this part

### CUDA at Scale for the Enterprise
// please fill in this part

### CUDA Advanced Libraries
// please fill in this part

### KLT Tracker with CUDA from scratch
// please fill in this part

## Prerequisites
  
### install cuda toolkit
```bash
# CUDA
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```

### Install OpenCV 4.7.0 with FFMPEG and CUDA support:

```bash
# Install FFMPEG development libraries
sudo apt update && sudo apt install -y libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libgtk2.0-dev libcanberra-gtk-module

# Download OpenCV and opencv_contrib
wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/4.7.0.zip \
&& wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/4.7.0.zip

# Extract both
unzip opencv.zip && unzip opencv_contrib.zip

cd opencv-4.7.0

# Configure and build OpenCV with FFMPEG and CUDA support
rm -rf build/ \
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
      -DBUILD_LIST=core,cudev,imgproc,imgcodecs,videoio,highgui,video \
&& cmake --build build/ --parallel $(nproc) && cmake --install build/
```
