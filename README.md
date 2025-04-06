# GPU_Programming_Specialization

```
# CUDA
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# OpenCV
export LD_LIBRARY_PATH=/home/lai/GPU_Programming_Specialization/opencv-4.7.0/install_opencv/lib:$LD_LIBRARY_PATH


```

## build and install opencv locally

```
wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/4.7.0.zip
unzip opencv.zip
cd opencv-4.7.0

cmake -S . -B build/ -GNinja \
  -DCMAKE_INSTALL_PREFIX=./install_opencv

cmake --build build --parallel $(nproc) && cmake --install build/
```
