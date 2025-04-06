IDIR=./
COMPILER=nvcc
COMPILER_FLAGS=-I$(IDIR) -I/usr/local/cuda/include -I/usr/local/cuda/lib64 -lcudart -lcuda --std c++17

.PHONY: clean build run

build-memory-allocation: memory_allocation.cu memory_allocation.h
	$(COMPILER) $(COMPILER_FLAGS) memory_allocation.cu -o memory_allocation.exe

build-memory-copy: memory_copy.cu memory_copy.h
	$(COMPILER) $(COMPILER_FLAGS) memory_copy.cu -o memory_copy.exe

build-broken-paged-pinned-memory-allocation: broken_paged_pinned_memory_allocation.cu broken_paged_pinned_memory_allocation.h
	$(COMPILER) $(COMPILER_FLAGS) broken_paged_pinned_memory_allocation.cu -o broken_paged_pinned_memory_allocation.exe

build-broken-mapped-memory-allocation: broken_mapped_memory_allocation.cu broken_mapped_memory_allocation.h
	$(COMPILER) $(COMPILER_FLAGS) broken_mapped_memory_allocation.cu -o broken_mapped_memory_allocation.exe

build: build-memory-allocation build-memory-copy build-broken-paged-pinned-memory-allocation build-broken-mapped-memory-allocation

clean:
	rm -f *.exe output*.txt

run-memory-allocation:
	./memory_allocation.exe $(ARGS)

run-memory-copy:
	./memory_copy.exe $(ARGS)

run-broken-paged-pinned-memory-allocation:
	./broken_paged_pinned_memory_allocation.exe $(ARGS)

run-broken-mapped-memory-allocation:
	./broken_mapped_memory_allocation.exe $(ARGS)

all: clean build