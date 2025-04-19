CXX = g++
CXXFLAGS = -std=c++17 -pthread -Wall -Wextra
TARGET = sleeping_barber
SRC = sleeping_barber.cpp

.PHONY: all clean run

all: clean build run

build: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)