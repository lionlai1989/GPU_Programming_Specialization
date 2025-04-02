IDIR=./
COMPILER=nvcc
COMPILER_FLAGS=-I$(IDIR) -I/usr/local/cuda/include -lcuda --std c++17

.PHONY: clean build run

build: calculation.cu calculation.h
	$(COMPILER) $(COMPILER_FLAGS) calculation.cu -o calculation.exe

clean:
	rm -f *.exe

run:
	./calculation.exe

all: clean build run
