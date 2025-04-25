CXX = nvcc
CXXFLAGS = -std=c++17 -O3
TARGET = main
SRC = main.cu

.PHONY: all clean run

all: clean build run

build: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

run:
	./$(TARGET)

all: clean build run
