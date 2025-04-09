# IDIR=./
CXX = nvcc

OPENCV_DIR := ../../../opencv-4.7.0/install_opencv

# Include and library paths
INCLUDES := -I$(OPENCV_DIR)/include/opencv4
LIBDIRS  := -L$(OPENCV_DIR)/lib

# List the OpenCV libraries you need to link
LIBS := -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

# CXXFLAGS += $(shell pkg-config --cflags --libs opencv4)
# LDFLAGS += $(shell pkg-config --libs --static opencv)

all: clean build

build: 
	$(CXX) sharedConstantMemoryLab.cu --std c++17 $(INCLUDES) $(LIBDIRS) $(LIBS) -o sharedConstantMemoryLab.exe -Wno-deprecated-gpu-targets -I/usr/local/cuda/include -lcuda
#	$(CXX) sharedConstantMemoryLab.cu --std c++17 `pkg-config opencv --cflags --libs` -o sharedConstantMemoryLab.exe -Wno-deprecated-gpu-targets $(CXXFLAGS) -I/usr/local/cuda/include -lcuda

run:
	./sharedConstantMemoryLab.exe $(ARGS)

clean:
	rm -f sharedConstantMemoryLab.exe