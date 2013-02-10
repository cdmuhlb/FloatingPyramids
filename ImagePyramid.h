#ifndef IMAGEPYRAMID_H_
#define IMAGEPYRAMID_H_

#include <cassert>
#include <vector_types.h>

class ImagePyramid {
 public:
  ImagePyramid(int nLevels, uint2 dim);
  ~ImagePyramid();

  int NLevels() const { return nLevels_; }

  uint2 Dim(int level) const {
    assert((level >= 0) && (level < nLevels_));
    return dims_[level];
  }

  int Size(int level) const {
    assert((level >= 0) && (level < nLevels_));
    return dims_[level].x * dims_[level].y;
  }

  float* GetLevel(int level) {
    assert((level >= 0) && (level < nLevels_));
    return d_levels_[level];
  }

  const float* GetLevel(int level) const {
    assert((level >= 0) && (level < nLevels_));
    return d_levels_[level];
  }

 private:
  const int nLevels_;
  uint2* dims_;
  float** d_levels_;
};

#endif  // IMAGEPYRAMID_H_

