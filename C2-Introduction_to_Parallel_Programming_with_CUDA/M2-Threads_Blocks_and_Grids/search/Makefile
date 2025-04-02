IDIR=./
COMPILER=nvcc
COMPILER_FLAGS=-I$(IDIR) -I/usr/local/cuda/include -lcuda --std c++17

.PHONY: clean build run

build: search.cu search.h
	$(COMPILER) $(COMPILER_FLAGS) search.cu -o search.exe

clean:
	rm -f search.exe output*.txt

run:
	./search.exe $(ARGS)

all: clean build run