# IDIR=./
CXX = nvcc

CXXFLAGS += $(shell pkg-config --cflags --libs opencv4)
LDFLAGS += $(shell pkg-config --libs --static opencv)

all: clean build

build: memory_analysis.cu memory_analysis.h
	$(CXX) memory_analysis.cu --std c++17 `pkg-config opencv --cflags --libs` -o memory_analysis.exe -Wno-deprecated-gpu-targets $(CXXFLAGS) -I/usr/local/cuda/include -lcuda

run:
	./memory_analysis.exe $(ARGS)

clean:
	rm -f memory_analysis.exe output*.txt output*.csv output*.dat output*.png