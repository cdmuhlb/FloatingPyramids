#include "ImagePyramid.h"

#include <cuda.h>
#include "utils.h"

ImagePyramid::ImagePyramid(const int nLevels, const uint2 dim)
    : nLevels_(nLevels) {
  dims_ = new uint2[nLevels_];
  d_levels_ = new float*[nLevels_];
  uint2 dimi = dim;
  for (int i=0; i<nLevels_; ++i) {
    checkCudaErrors(cudaMalloc(&d_levels_[i], dimi.x*dimi.y*sizeof(float)));
    dims_[i] = dimi;
    dimi.x /= 2;
    dimi.y /= 2;
  }
}

ImagePyramid::~ImagePyramid() {
  for (int i=0; i<nLevels_; ++i) {
    cudaFree(d_levels_[i]);
  }
  delete[] d_levels_;
  delete[] dims_;
}

