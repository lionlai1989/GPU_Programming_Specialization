CXX = nvcc
CXXFLAGS = -std=c++17 -O3
TARGET = vector_add_sub
SRC = vector_add_sub.cu

.PHONY: all clean run

all: clean build run

build: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)