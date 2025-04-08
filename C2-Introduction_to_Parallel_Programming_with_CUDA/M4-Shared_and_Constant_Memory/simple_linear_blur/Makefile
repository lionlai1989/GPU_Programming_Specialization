# IDIR=./
CXX = nvcc

CXXFLAGS += $(shell pkg-config --cflags --libs opencv4)
LDFLAGS += $(shell pkg-config --libs --static opencv)

all: clean build

build: 
	$(CXX) simpleLinearBlurFilter.cu --std c++17 `pkg-config opencv --cflags --libs` -o simpleLinearBlurFilter.exe -Wno-deprecated-gpu-targets $(CXXFLAGS) -I/usr/local/cuda/include -lcuda

run:
	./simpleLinearBlurFilter.exe $(ARGS)

clean:
	rm -f simpleLinearBlurFilter.exe