cmake_minimum_required(VERSION 3.10)
project(KLT_Tracker_CUDA LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set compiler flags based on build type
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 86)  # Adjust this based on your target GPU
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Set OpenCV paths
set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../opencv-4.7.0/install_opencv/lib/cmake/opencv4")
if(NOT EXISTS ${OpenCV_DIR})
    message(FATAL_ERROR "OpenCV directory not found at ${OpenCV_DIR}")
endif()

# Find required packages
find_package(OpenCV REQUIRED COMPONENTS 
    core
    imgproc
    highgui
    video
    cudaarithm
    cudafilters
    cudaimgproc
    cudawarping
)
find_package(CUDA REQUIRED)

# Add executables for both tracker implementations
add_executable(tracker_basic src/tracker_basic.cpp)
add_executable(tracker_opencv src/tracker_opencv.cpp)

# Common OpenCV configuration for both executables
foreach(target tracker_basic tracker_opencv)
    # Link OpenCV libraries using modern CMake practices
    target_link_libraries(${target} PRIVATE
        ${OpenCV_LIBS}
        opencv_core
        opencv_highgui
        opencv_imgproc
        opencv_video
    )

    # Include OpenCV directories
    target_include_directories(${target} PRIVATE ${OpenCV_INCLUDE_DIRS})
endforeach()

# Enable testing
enable_testing()

# Add installation rules
install(TARGETS tracker_basic tracker_opencv
    RUNTIME DESTINATION bin
)