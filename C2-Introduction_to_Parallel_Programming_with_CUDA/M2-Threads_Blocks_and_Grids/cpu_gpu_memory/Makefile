IDIR=./
COMPILER=nvcc
COMPILER_FLAGS=-I$(IDIR) -I/usr/local/cuda/include -I/usr/local/cuda/lib64 -lcudart -lcuda --std c++17

.PHONY: clean build run

build: cpu_gpu_memory.cu cpu_gpu_memory.h
	$(COMPILER) $(COMPILER_FLAGS) cpu_gpu_memory.cu -o cpu_gpu_memory.exe

clean:
	rm -f cpu_gpu_memory.exe

run:
	./cpu_gpu_memory.exe

all: clean build run
