cmake_minimum_required(VERSION 3.10)
project(KLT_Tracker_CPP)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set OpenCV paths
set(OpenCV_DIR "/Users/lionlai/GPU_Programming_Specialization/opencv-4.7.0/install_opencv/lib/cmake/opencv4")
set(OpenCV_INCLUDE_DIRS "/Users/lionlai/GPU_Programming_Specialization/opencv-4.7.0/install_opencv/include/opencv4")
set(OpenCV_LIBS_DIR "/Users/lionlai/GPU_Programming_Specialization/opencv-4.7.0/install_opencv/lib")

# Find OpenCV package with required components
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc video)

# Add executables for both tracker implementations
add_executable(klt_tracker_basic tracker_basic.cpp)
add_executable(klt_tracker_opencv tracker_opencv.cpp)

# Common OpenCV configuration for both executables
foreach(target klt_tracker_basic klt_tracker_opencv)
    # Link OpenCV libraries
    target_link_directories(${target} PRIVATE ${OpenCV_LIBS_DIR})
    target_link_libraries(${target} 
        opencv_core
        opencv_highgui
        opencv_imgproc
        opencv_video
    )

    # Include OpenCV directories
    target_include_directories(${target} PRIVATE ${OpenCV_INCLUDE_DIRS})
endforeach()