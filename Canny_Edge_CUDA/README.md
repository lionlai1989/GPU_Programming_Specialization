# Canny Edge CUDA

![Video Comparison](combine.gif)

## Environment Setup

The following instructions only work for Linux/Ubuntu systems.

### Prerequisites
- CUDA Toolkit 11.8 (or compatible version)
  - Make sure nvcc is in your PATH
  - You can verify by running `which nvcc`
  - If not found, add CUDA to your PATH:
    ```bash
    export PATH=/usr/local/cuda-11.8/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
    ```

### Setup Steps

1. Install OpenCV 4.7.0 with FFMPEG and CUDA support:

```bash
# Install FFMPEG development libraries
sudo apt update && sudo apt install -y libavcodec-dev libavformat-dev libavutil-dev libswscale-dev

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
      -DBUILD_opencv_hdf=OFF \
      -DBUILD_TESTS=OFF \
      -DBUILD_PERF_TESTS=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DOPENCV_GENERATE_PKGCONFIG=ON \
      -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.7.0/modules \
      -DBUILD_LIST=core,cudev,imgproc,imgcodecs,videoio \
&& cmake --build build/ --parallel $(nproc) && cmake --install build/
```

### Build and Run

#### Build the project: `rm -rf build/ && cmake -S . -B build/ && cmake --build build/ -j 4`

#### Run CLI with arguments: `./build/main.exe -i data/me.mp4 -o output/canny_edge.mp4`


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
