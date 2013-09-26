# User settings
# =============

# Compute capability of target GPU (passed to '-arch' flag)
CUDA_ARCH = sm_30

# Location of the CUDA Toolkit
CUDA_PATH = /usr/local/cuda

# Compilers and flags
CXX = g++
CXXFLAGS = -O3 -march=native -Wall -Wextra
NVCC = nvcc
NVCCFLAGS = -O3 -arch=$(CUDA_ARCH)

# Library paths
#   Example LIB: -L /usr/lib
#   Example INCLUDE: -I /usr/include
OPENCV_LIB =
OPENCV_INCLUDE =


# Internal settings
# =================

# Includes and libraries
CUDA_INCLUDE = -I $(CUDA_PATH)/include
OPENCV_LIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui
INCLUDES = $(OPENCV_INCLUDE) $(CUDA_INCLUDE)
LIBRARIES = $(OPENCV_LIB) $(OPENCV_LIBS)


# Targets
# =======

all: bin/pyramids

bin/pyramids: bin build/main.o build/PyramidMaker.o build/ImagePyramid.o build/PyramidKernels.o
	$(NVCC) build/main.o build/PyramidMaker.o build/ImagePyramid.o build/PyramidKernels.o -o $@ $(LIBRARIES) $(NVCCFLAGS)

bin:
	mkdir -p bin

build/main.o: build src/main.cc src/timer.h src/utils.h src/PyramidMaker.h
	$(CXX) -c src/main.cc -o $@ $(INCLUDES) $(CXXFLAGS)

build/PyramidMaker.o: build src/PyramidMaker.cc src/PyramidMaker.h src/utils.h src/ImagePyramid.h src/PyramidKernels.h
	$(CXX) -c src/PyramidMaker.cc -o $@ $(INCLUDES) $(CXXFLAGS)

build/ImagePyramid.o: build src/ImagePyramid.cc src/ImagePyramid.h src/utils.h
	$(CXX) -c src/ImagePyramid.cc -o $@ $(INCLUDES) $(CXXFLAGS)

build/PyramidKernels.o: build src/PyramidKernels.cu src/PyramidKernels.h src/utils.h
	$(NVCC) -c src/PyramidKernels.cu -o $@ $(NVCCFLAGS)

build:
	mkdir -p build

clean:
	rm -f build/main.o build/PyramidMaker.o build/ImagePyramid.o build/PyramidKernels.o bin/pyramids
