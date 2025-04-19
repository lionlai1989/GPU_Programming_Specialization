CXX = g++
CXXFLAGS = -std=c++17 -pthread -Wall -Wextra
TARGET = main
SRC = main.cpp

.PHONY: all clean run

all: clean build run

build: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)