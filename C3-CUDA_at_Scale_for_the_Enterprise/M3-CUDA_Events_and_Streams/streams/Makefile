IDIR=./
COMPILER=nvcc
COMPILER_FLAGS=-I$(IDIR) -I/usr/local/cuda/include -I/usr/local/cuda/lib64 -lcudart -lcuda --std c++17

.PHONY: clean build run

build: *.cu
	$(COMPILER) $(COMPILER_FLAGS) *.cu -o streams.exe

clean:
	rm -f streams.exe

run:
	./streams.exe

all: clean build run
