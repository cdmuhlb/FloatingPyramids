# This file is derived from materials provided by Udacity course CS344:
# Introduction to Parallel Programming (https://github.com/udacity/cs344).

CXX=g++
NVCC=nvcc
CUDA_ARCH=sm_30

###################################
# These are the default install   #
# locations on most linux distros #
###################################

OPENCV_LIBPATH=/usr/lib
OPENCV_INCLUDEPATH=/usr/include

###################################################
# On Macs the default install locations are below #
###################################################

#OPENCV_LIBPATH=/usr/local/lib
#OPENCV_INCLUDEPATH=/usr/local/include

OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui

CUDA_INCLUDEPATH=/usr/local/cuda/include

######################################################
# On Macs the default install locations are below    #
######################################################

#CUDA_INCLUDEPATH=/usr/local/cuda/include
#CUDA_LIBPATH=/usr/local/cuda/lib

# Production profile
NVCC_OPTS=-O3 -arch=$(CUDA_ARCH) -Xcompiler -Wall -Xcompiler -Wextra
LINK_OPTS=-O3 -arch=$(CUDA_ARCH)
GCC_OPTS=-O3 -march=native -Wall -Wextra

# Debugging profile
#NVCC_OPTS=-arch=$(CUDA_ARCH) -Xcompiler -Wall -Xcompiler -Wextra -g -G
#LINK_OPTS=-arch=$(CUDA_ARCH) -g -G
#GCC_OPTS=-Wall -Wextra -g

pyramids: main.o PyramidMaker.o ImagePyramid.o PyramidKernels.o Makefile
	$(NVCC) -o pyramids main.o PyramidMaker.o ImagePyramid.o PyramidKernels.o -L $(OPENCV_LIBPATH) $(OPENCV_LIBS) $(LINK_OPTS)

main.o: main.cc timer.h utils.h PyramidMaker.h
	$(CXX) -c main.cc $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

PyramidMaker.o: PyramidMaker.cc PyramidMaker.h utils.h ImagePyramid.h PyramidKernels.h
	$(CXX) -c PyramidMaker.cc $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

ImagePyramid.o: ImagePyramid.cc ImagePyramid.h utils.h
	$(CXX) -c ImagePyramid.cc $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

PyramidKernels.o: PyramidKernels.cu PyramidKernels.h utils.h
	$(NVCC) -c PyramidKernels.cu $(NVCC_OPTS)

clean:
	rm -f *.o pyramids
