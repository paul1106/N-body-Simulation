# Makefile for HIP N-Body Simulation (hw5)

CXX = hipcc
CXXFLAGS = -O3 -std=c++17 --offload-arch=gfx908:sramecc+:xnack- -munsafe-fp-atomics
CXXFLAGS += -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false
TARGET = hw5

all: $(TARGET)

$(TARGET): hw5.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) hw5.cpp

clean:
	rm -f $(TARGET)

test: $(TARGET)
	./$(TARGET) testcases/b20.in output.txt
	cat output.txt

.PHONY: all clean test
