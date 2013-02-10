#ifndef PYRAMIDMAKER_H_
#define PYRAMIDMAKER_H_

#include <string>
#include <vector_types.h>

class ImagePyramid;

void MakePyramids(const std::string& inputFilename,
                  const std::string& outputPrefix);

void GlobalLaplacianCompression(const float* d_in, uint2 dim, float* d_out);
void LocalLaplacianCompression(const float* d_in, uint2 dim, float* d_out);
void FastLocalLaplacianCompression(const float* d_in, uint2 dim, float* d_out);

void ConstructGaussianPyramid(const float* d_image, ImagePyramid& gPyramid);

void ConstructLaplacianPyramid(const ImagePyramid& gPyramid,
                               ImagePyramid& lPyramid);

void ConstructPartialGaussianPyramid(ImagePyramid& gPyramid, int maxLevel);

void ConstructPartialLaplacianPyramid(const ImagePyramid& gPyramid,
                                      int maxLevel,
                                      ImagePyramid& lPyramid);

void CollapseLaplacianPyramid(const ImagePyramid& lPyramid,
                              ImagePyramid& gPyramid, float* d_image);

void WriteDeviceImage(const float* d_image, uint2 dim,
                      const std::string& filename);

#endif  // PYRAMIDMAKER_H_

