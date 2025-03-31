IDIR=./
STD :=c++14
CXX=g++
CXXFLAGS=-I$(IDIR) -std=$(STD) -pthread

.PHONY: clean build run_all

build_threading_example: thread_example.cpp thread_example.h
	$(CXX) -o thread_example.exe $(CXXFLAGS) thread_example.cpp

build_mutex_example: mutex_example.cpp mutex_example.h
	$(CXX) -o mutex_example.exe $(CXXFLAGS) mutex_example.cpp

build_atomic_example: atomic_example.cpp atomic_example.h
	$(CXX) -o atomic_example.exe $(CXXFLAGS) atomic_example.cpp

build_future_example: future_example.cpp future_example.h
	$(CXX) -o future_example.exe $(CXXFLAGS) future_example.cpp

build: build_threading_example build_mutex_example build_atomic_example build_future_example

clean:
	rm -f *.exe

run_threading_example:
	./thread_example.exe

run_mutex_example:
	./mutex_example.exe

run_atomic_example:
	./atomic_example.exe

run_future_example:
	./future_example.exe

run_all: run_threading_example run_mutex_example run_atomic_example run_future_example

all: clean build run_all

