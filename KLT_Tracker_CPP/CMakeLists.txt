cmake_minimum_required(VERSION 3.10)
project(KLT_Tracker_CPP)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set OpenCV paths
set(OpenCV_DIR "/home/lai/GPU_Programming_Specialization/opencv-4.7.0/install_opencv/lib/cmake/opencv4")
set(OpenCV_INCLUDE_DIRS "/home/lai/GPU_Programming_Specialization/opencv-4.7.0/install_opencv/include/opencv4")
set(OpenCV_LIBS_DIR "/home/lai/GPU_Programming_Specialization/opencv-4.7.0/install_opencv/lib")

# Find OpenCV package with required components
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc video)

# Add executable
add_executable(KLT_Tracker_CPP main.cpp)

# Link OpenCV libraries
target_link_directories(KLT_Tracker_CPP PRIVATE ${OpenCV_LIBS_DIR})
target_link_libraries(KLT_Tracker_CPP 
    opencv_core
    opencv_highgui
    opencv_imgproc
    opencv_video
)

# Include OpenCV directories
target_include_directories(KLT_Tracker_CPP PRIVATE ${OpenCV_INCLUDE_DIRS})