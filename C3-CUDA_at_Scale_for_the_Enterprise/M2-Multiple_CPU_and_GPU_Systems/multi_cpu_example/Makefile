IDIR=./
COMPILER=nvcc
COMPILER_FLAGS=-I$(IDIR) -I/usr/local/cuda/include -lcuda --std c++17

.PHONY: clean build run

build: multi_cpu.cu multi_cpu.h
	$(COMPILER) $(COMPILER_FLAGS) multi_cpu.cu -o multi_cpu.exe -Wno-deprecated-gpu-targets

clean:
	rm -f multi_cpu.exe

run: run_producer_consumer_a run_producer_consumer_b run_multicpu_cuda

run_multicpu_cuda:
	./multi_cpu.exe

run_producer_consumer_a:
	python3 producer_consumer.py a 128

run_producer_consumer_b:
	python3 producer_consumer.py b 128

all: clean build run
