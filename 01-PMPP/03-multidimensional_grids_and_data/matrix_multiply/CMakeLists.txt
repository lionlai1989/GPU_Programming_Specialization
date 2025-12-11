cmake_minimum_required(VERSION 3.10)

# First find CUDA before project() to ensure proper compiler detection
find_program(CMAKE_CUDA_COMPILER nvcc
    PATHS
    /usr/local/cuda-12.4/bin
    /usr/local/cuda/bin
    ENV PATH
    NO_DEFAULT_PATH
)


if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA compiler (nvcc) not found. Please ensure CUDA is installed and nvcc is in your PATH.")
endif()

project(matrix_multiply LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 86)  # Adjust this based on your GPU architecture

# Set OpenCV paths
set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../opencv-4.10.0/install_opencv/lib/cmake/opencv4")

# Find OpenCV package
find_package(OpenCV REQUIRED COMPONENTS 
    core
    imgproc
    videoio
    cudaarithm
    cudafilters
    cudaimgproc
    cudawarping
)

# Set C++ and CUDA standards
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Add executable
add_executable(matrix_multiply main.cu)

# Link libraries
target_link_libraries(matrix_multiply 
    ${OpenCV_LIBS}
    cudart
)

# Include directories
target_include_directories(matrix_multiply PRIVATE ${OpenCV_INCLUDE_DIRS})
