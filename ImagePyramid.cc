#include "ImagePyramid.h"

#include <cuda.h>
#include "utils.h"

ImagePyramid::ImagePyramid(const int nLevels, const uint2 dim)
    : nLevels_(nLevels), dim_(dim) {
  d_gLevels_ = new float*[nLevels_];
  d_lLevels_ = new float*[nLevels_-1];
  int nxi = dim_.x;
  int nyi = dim_.y;
  for (int i=0; i<nLevels_-1; ++i) {
    checkCudaErrors(cudaMalloc(&d_gLevels_[i], nxi*nyi*sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_lLevels_[i], nxi*nyi*sizeof(float)));
    nxi /= 2;
    nyi /= 2;
  }
  checkCudaErrors(cudaMalloc(&d_gLevels_[nLevels_-1],
      nxi*nyi*sizeof(float)));
}

ImagePyramid::~ImagePyramid() {
  for (int i=0; i<nLevels_-1; ++i) {
    cudaFree(d_lLevels_[i]);
    cudaFree(d_gLevels_[i]);
  }
  cudaFree(d_gLevels_[nLevels_-1]);
  delete[] d_lLevels_;
  delete[] d_gLevels_;
}

