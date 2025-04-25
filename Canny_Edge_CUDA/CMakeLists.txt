cmake_minimum_required(VERSION 3.10)

# First find CUDA before project() to ensure proper compiler detection
find_program(CUDA_NVCC_EXECUTABLE nvcc
    PATHS
    /usr/local/cuda-11.8/bin
    /usr/local/cuda/bin
    ENV PATH
    NO_DEFAULT_PATH
)

if(NOT CUDA_NVCC_EXECUTABLE)
    message(FATAL_ERROR "CUDA compiler (nvcc) not found. Please ensure CUDA is installed and nvcc is in your PATH.")
endif()

# Enable compile commands export
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(Canny_Edge_CUDA LANGUAGES CXX CUDA)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 86)  # Adjust this based on your GPU architecture

# Set OpenCV paths
set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../opencv-4.7.0/install_opencv/lib/cmake/opencv4")

# Find required packages
find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR} NO_DEFAULT_PATH)
find_package(CUDA REQUIRED)

# Set C++ and CUDA standards
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Add executables
add_executable(canny_cuda src/canny_cuda.cu)
add_executable(canny_opencv src/canny_opencv.cpp)

# Link libraries for CUDA version
target_link_libraries(canny_cuda 
    ${OpenCV_LIBS} 
    cudart 
    nppc 
    nppial 
    nppisu 
    nppig 
    nppicc 
    nppif
    nppidei
    nppist
)

# Link libraries for OpenCV version
target_link_libraries(canny_opencv ${OpenCV_LIBS})

# Include directories
target_include_directories(canny_cuda PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(canny_opencv PRIVATE ${OpenCV_INCLUDE_DIRS})

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Add custom targets for running the programs
add_custom_target(run_cuda
    COMMAND canny_cuda
    DEPENDS canny_cuda
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)

add_custom_target(run_opencv
    COMMAND canny_opencv
    DEPENDS canny_opencv
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)
