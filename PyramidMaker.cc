#include "PyramidMaker.h"

#include <cstdlib>
#include <sstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "ImagePyramid.h"
#include "PyramidKernels.h"

void MakePyramids(const std::string& inputFilename,
                  const std::string& outputPrefix) {
  // Load input image
  cv::Mat origColor;
  {
    cv::Mat tmpImage;
    tmpImage = cv::imread(inputFilename.c_str(), CV_LOAD_IMAGE_COLOR);
    if (tmpImage.empty()) {
      std::cerr << "Couldn't open file: " << inputFilename << std::endl;
      abort();
    }
    cv::cvtColor(tmpImage, origColor, CV_BGR2RGBA);
  }

  const int rows = origColor.rows;
  const int cols = origColor.cols;
  const uint2 dim = {cols, rows};
  const int origSize = rows*cols;

  // Convert input image to greyscale
  float* d_greyImage;
  {
    uchar4* d_colorImage;
    checkCudaErrors(cudaMalloc(&d_colorImage, origSize*sizeof(uchar4)));
    checkCudaErrors(cudaMalloc(&d_greyImage, origSize*sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_colorImage,
        (uchar4*)origColor.ptr<unsigned char>(0), origSize*sizeof(uchar4),
        cudaMemcpyHostToDevice));
    ConvertToGreyscale(d_colorImage, dim, d_greyImage);
    cudaFree(d_colorImage);
  }

  /* Construct image pyramid */
  const int nLevels = 9;
  ImagePyramid pyramid(nLevels, dim);
  checkCudaErrors(cudaMemcpy(pyramid.GLevel(0), d_greyImage,
      origSize*sizeof(float), cudaMemcpyDeviceToDevice));

  int nx = cols;
  int ny = rows;
  for (int level=0; level<pyramid.NLevels()-1; ++level) {
    const uint2 thisDim = {nx, ny};
    ComputeNextGLevel(pyramid.GLevel(level), thisDim, pyramid.GLevel(level+1));
    ComputeThisLLevel(pyramid.GLevel(level), pyramid.GLevel(level+1),
        thisDim, pyramid.LLevel(level));
    nx /= 2;
    ny /= 2;
  }
  const int lastNx = nx;
  const int lastNy = ny;

  /* Filter Laplacian coefficients */
  float* d_fwork;
  checkCudaErrors(cudaMalloc(&d_fwork, origSize*sizeof(float)));
  nx = cols;
  ny = rows;
  for (int level=0; level<pyramid.NLevels()-1; ++level) {
    const uint2 thisDim = {nx, ny};
    FilterThisLLevel(pyramid.LLevel(level), thisDim, 1.0f/64.0f, d_fwork);
    checkCudaErrors(cudaMemcpy(pyramid.LLevel(level), d_fwork,
        nx*ny*sizeof(float), cudaMemcpyDeviceToDevice));
    nx /= 2;
    ny /= 2;
  }
  cudaFree(d_fwork);

  /* Reconstruct image from pyramid */
  float* d_gThis;
  float* d_gNext;
  checkCudaErrors(cudaMalloc(&d_gThis, origSize*sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_gNext, origSize*sizeof(float)));

  nx = lastNx;
  ny = lastNy;
  checkCudaErrors(cudaMemcpy(d_gThis, pyramid.GLevel(pyramid.NLevels()-1),
      nx*ny*sizeof(float), cudaMemcpyDeviceToDevice));
  for (int level=pyramid.NLevels()-2; level>=0; --level) {
    float* swp = d_gNext;
    d_gNext = d_gThis;
    d_gThis = swp;
    nx *= 2;
    ny *= 2;
    const uint2 thisDim = {nx, ny};
    ReconstructThisGLevel(d_gNext, pyramid.LLevel(level), thisDim, d_gThis);
  }
  // final result in d_gThis

  /* Postprocess */
  ShiftAndStretch(d_gThis, dim, 1.75f, 0.3f);

  /* Write to disk */
  {
    unsigned char* d_imageBytes;
    checkCudaErrors(cudaMalloc(&d_imageBytes, origSize*sizeof(unsigned char)));
    ClampToBytes(d_gThis, dim, d_imageBytes);
    cv::Mat outImage;
    outImage.create(rows, cols, CV_8UC1);
    checkCudaErrors(cudaMemcpy(outImage.ptr<unsigned char>(0), d_imageBytes,
        origSize*sizeof(unsigned char), cudaMemcpyDeviceToHost));
    cudaFree(d_imageBytes);
    std::ostringstream outName;
    outName << outputPrefix << "_G0prime.png";
    cv::imwrite(outName.str().c_str(), outImage);
  }

  cudaFree(d_gNext);
  cudaFree(d_gThis);

  cudaFree(d_greyImage);
}

