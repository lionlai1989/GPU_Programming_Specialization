# IDIR=./
CXX = nvcc

CXXFLAGS += $(shell pkg-config --cflags --libs opencv4)
LDFLAGS += $(shell pkg-config --libs --static opencv)

all: clean build

build: 
	$(CXX) sharedConstantMemoryLab.cu --std c++17 `pkg-config opencv --cflags --libs` -o sharedConstantMemoryLab.exe -Wno-deprecated-gpu-targets $(CXXFLAGS) -I/usr/local/cuda/include -lcuda

run:
	./sharedConstantMemoryLab.exe $(ARGS)

clean:
	rm -f sharedConstantMemoryLab.exe