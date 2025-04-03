# IDIR=./
CXX = nvcc

CXXFLAGS += $(shell pkg-config --cflags --libs opencv4)
LDFLAGS += $(shell pkg-config --libs --static opencv)

all: clean build

build: 
	$(CXX) convertRGBToGrey.cu --std c++17 `pkg-config opencv --cflags --libs` -o convertRGBToGrey.exe -Wno-deprecated-gpu-targets $(CXXFLAGS) -I/usr/local/cuda/include -lcuda

run:
	./convertRGBToGrey.exe $(ARGS)

clean:
	rm -f convertRGBToGrey.exe