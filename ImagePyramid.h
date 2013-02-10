#ifndef IMAGEPYRAMID_H_
#define IMAGEPYRAMID_H_

#include <cassert>
#include <cstddef>
#include <vector_types.h>

class ImagePyramid {
 public:
  ImagePyramid(int nLevels, uint2 dim);
  ~ImagePyramid();

  int NLevels() const { return nLevels_; }
  uint2 Dim() const { return dim_; }
  float* GLevel(int i) {
    assert((i >= 0) && (i < nLevels_));
    return d_gLevels_[i];
  }
  float* LLevel(int i) {
    assert((i >= 0) && (i < nLevels_-1));
    return d_lLevels_[i];
  }

 private:
  const int nLevels_;
  const uint2 dim_;
  float** d_gLevels_;
  float** d_lLevels_;
};

#endif  // IMAGEPYRAMID_H_

