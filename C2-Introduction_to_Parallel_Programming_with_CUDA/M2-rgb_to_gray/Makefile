CXX = nvcc
CXXFLAGS = -std=c++17 -O3
TARGET = main
SRC = main.cu

# OpenCV install path
OPENCV_DIR := ../../opencv-4.7.0/install_opencv

# Include and library paths
INCLUDES := -I$(OPENCV_DIR)/include/opencv4
LIBDIRS  := -L$(OPENCV_DIR)/lib

# List the OpenCV libraries you need to link
LIBS := -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

all: clean build run

build: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(SRC) $(INCLUDES) $(LIBDIRS) $(LIBS) -o $(TARGET) -Wno-deprecated-gpu-targets

run: $(TARGET)
	LD_LIBRARY_PATH=$(OPENCV_DIR)/lib:$(LD_LIBRARY_PATH) ./$(TARGET)

clean:
	rm -f $(TARGET)