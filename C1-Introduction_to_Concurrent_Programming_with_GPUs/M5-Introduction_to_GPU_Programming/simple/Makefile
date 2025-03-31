IDIR=./
COMPILER=REPLACE_WITH_NVIDIA_CROSS_COMPILER
COMPILER_FLAGS=-I$(IDIR) -I/usr/local/cuda/include -lcuda --std c++17

.PHONY: clean build run

build: simple.cu simple.h
	$(COMPILER) $(COMPILER_FLAGS) simple.cu -o simple.exe

clean:
	rm -f simple.exe

run:
	./simple.exe

all: clean build run
