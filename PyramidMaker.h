#ifndef PYRAMIDMAKER_H_
#define PYRAMIDMAKER_H_

#include <string>

class ImagePyramid;

void MakePyramids(const std::string& inputFilename,
                  const std::string& outputPrefix);

void ConstructGaussianPyramid(const float* d_image, ImagePyramid& gPyramid);

void ConstructLaplacianPyramid(const ImagePyramid& gPyramid,
                               ImagePyramid& lPyramid);

void CollapseLaplacianPyramid(const ImagePyramid& lPyramid,
                              ImagePyramid& gPyramid, float* d_image);

#endif  // PYRAMIDMAKER_H_

