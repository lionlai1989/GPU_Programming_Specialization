# Canny Edge CUDA

![Video Comparison](combine.gif)

## Environment Setup

### Build and Run

#### Build the project: `rm -rf build/ && cmake -S . -B build/ && cmake --build build/ -j 4`

#### Run CLI with arguments:
```
./build/canny_cuda
./build/canny_opencv
```

## Profiling CUDA
```
nsys profile --trace=cuda,osrt,nvtx ./build/canny_cuda

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
