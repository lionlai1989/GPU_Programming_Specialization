IDIR=./
STD :=c++14
CXX=g++
CXXFLAGS=-I$(IDIR) -Wall -Wextra -pedantic-errors -std=$(STD) -O2 -pthread

build: *.cpp
	$(CXX) -o producer_consumer $(CXXFLAGS) producer_consumer.cpp

.PHONY: clean

clean:
	rm -f producer_consumer

run:
	./producer_consumer $(ARGS)