#include "PyramidMaker.h"

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
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

  float* d_initialImage;
  float* d_finalImage;
  checkCudaErrors(cudaMalloc(&d_initialImage, origSize*sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_finalImage, origSize*sizeof(float)));

  // Convert input image to greyscale
  {
    uchar4* d_colorImage;
    checkCudaErrors(cudaMalloc(&d_colorImage, origSize*sizeof(uchar4)));

    checkCudaErrors(cudaMemcpy(d_colorImage,
        (uchar4*)origColor.ptr<unsigned char>(0), origSize*sizeof(uchar4),
        cudaMemcpyHostToDevice));
    ConvertToGreyscale(d_colorImage, dim, d_initialImage);
    cudaFree(d_colorImage);
  }

  // Pyramid work
  //GlobalLaplacianCompression(d_initialImage, dim, d_finalImage);
  //LocalLaplacianCompression(d_initialImage, dim, d_finalImage);
  FastLocalLaplacianCompression(d_initialImage, dim, d_finalImage);

  // Postprocess
  //ShiftAndStretch(d_finalImage, dim, 1.75f, 0.2f);

  // Write to disk
  WriteDeviceImage(d_finalImage, dim, outputPrefix + "_final.png");
  //WriteDeviceImage(d_initialImage, dim, outputPrefix + "_initial.png");

  cudaFree(d_finalImage);
  cudaFree(d_initialImage);
}

void GlobalLaplacianCompression(const float* const d_in, const uint2 dim,
                                float* const d_out) {
  const int nLevels = 9;
  ImagePyramid gPyramid(nLevels, dim);
  ImagePyramid lPyramid(nLevels-1, dim);

  // Construct image pyramid
  ConstructGaussianPyramid(d_in, gPyramid);
  ConstructLaplacianPyramid(gPyramid, lPyramid);

  // Filter Laplacian coefficients
  float* d_fwork;
  checkCudaErrors(cudaMalloc(&d_fwork, lPyramid.Size(0)*sizeof(float)));
  for (int level=0; level<lPyramid.NLevels(); ++level) {
    FilterThisLLevel(lPyramid.GetLevel(level), lPyramid.Dim(level),
        1.0f/32.0f, d_fwork);
    checkCudaErrors(cudaMemcpy(lPyramid.GetLevel(level), d_fwork,
        lPyramid.Size(level)*sizeof(float), cudaMemcpyDeviceToDevice));
  }
  cudaFree(d_fwork);

  // Collapse pyramid
  CollapseLaplacianPyramid(lPyramid, gPyramid, d_out);
}

void LocalLaplacianCompression(const float* const d_in, const uint2 dim,
                               float* const d_out) {
  const int nLevels = 8;
  ImagePyramid gPyramid(nLevels, dim);
  ImagePyramid lPyramid(nLevels-1, dim);

  // Construct image pyramid
  ConstructGaussianPyramid(d_in, gPyramid);

  ImagePyramid gTmpP(nLevels, dim);
  ImagePyramid lTmpP(nLevels-1, dim);

  for (int level=0; level<lPyramid.NLevels(); ++level) {
    for (int idx=0; idx<gPyramid.Size(level); ++idx) {
      float g0;
      checkCudaErrors(cudaMemcpy(&g0, &gPyramid.GetLevel(level)[idx],
          sizeof(float), cudaMemcpyDeviceToHost));
      RemapImage(d_in, dim, g0, 0.25, 1.0f, 0.25f, gTmpP.GetLevel(0)); 

      ConstructPartialGaussianPyramid(gTmpP, level+1);
      ConstructPartialLaplacianPyramid(gTmpP, level, lTmpP);
      checkCudaErrors(cudaMemcpy(&lPyramid.GetLevel(level)[idx],
          &lTmpP.GetLevel(level)[idx], sizeof(float),
          cudaMemcpyDeviceToDevice));
      if (idx%1000 == 0) std::cout << "idx: " << idx << std::endl;
    }
  }

  // Collapse pyramid
  CollapseLaplacianPyramid(lPyramid, gPyramid, d_out);
}


void FastLocalLaplacianCompression(const float* const d_in, const uint2 dim,
                                   float* const d_out) {
  const int nLevels = 9;
  ImagePyramid gPyramid(nLevels, dim);
  ImagePyramid lPyramid(nLevels-1, dim);

  const int nGammas = 24;
  ImagePyramid** lps = new ImagePyramid*[nGammas];
  ImagePyramid gTmpP(nLevels, dim);
  for (int i=0; i<nGammas; ++i) {
    lps[i] = new ImagePyramid(nLevels-1, dim);
    const float gamma = i / (nGammas - 1.0f);
    RemapImage(d_in, dim, gamma, 1.0f, 0.25, 0.04, gTmpP.GetLevel(0));
    ConstructPartialGaussianPyramid(gTmpP, nLevels-1);
    ConstructLaplacianPyramid(gTmpP, *lps[i]);
  }

  ConstructGaussianPyramid(d_in, gPyramid);
  float** lpsByGamma = new float*[nGammas];
  float** d_lpsByGamma;
  checkCudaErrors(cudaMalloc(&d_lpsByGamma, nGammas*sizeof(float*)));
  for (int level=0; level<nLevels-1; ++level) {
    for (int i=0; i<nGammas; ++i) {
      lpsByGamma[i] = lps[i]->GetLevel(level);
    }
    checkCudaErrors(cudaMemcpy(d_lpsByGamma, lpsByGamma,
        nGammas*sizeof(float*), cudaMemcpyHostToDevice));
    ApplyLLFilter(gPyramid.GetLevel(level), gPyramid.Dim(level), d_lpsByGamma,
                  nGammas, lPyramid.GetLevel(level));
  }
  cudaFree(d_lpsByGamma);
  delete[] lpsByGamma;

  for (int i=0; i<nGammas; ++i) {
    delete lps[i];
  }
  delete[] lps;

  // Mix pyramids
  /*ImagePyramid origLP(nLevels-1, dim);
  ConstructLaplacianPyramid(gPyramid, origLP);
  for (int level=0; level<lPyramid.NLevels(); ++level) {
    const float wb = level / (nLevels - 1.0f);
    const float wa = (1.0f - wa);
    WeightedAverage(lPyramid.GetLevel(level), origLP.GetLevel(level),
        lPyramid.Dim(level), wa, wb);
  }*/
  //WeightedAverage(lPyramid.GetLevel(0), origLP.GetLevel(0),
  //    lPyramid.Dim(0), 0.0f, 1.0f);

  // Collapse pyramid
  CollapseLaplacianPyramid(lPyramid, gPyramid, d_out);
}

void ConstructGaussianPyramid(const float* const d_image,
                              ImagePyramid& gPyramid) {
  checkCudaErrors(cudaMemcpy(gPyramid.GetLevel(0), d_image,
      gPyramid.Size(0)*sizeof(float), cudaMemcpyDeviceToDevice));
  for (int level=0; level<gPyramid.NLevels()-1; ++level) {
    ComputeNextGLevel(gPyramid.GetLevel(level), gPyramid.Dim(level),
        gPyramid.GetLevel(level+1));
  }
}

void ConstructLaplacianPyramid(const ImagePyramid& gPyramid, 
                               ImagePyramid& lPyramid) {
  assert(gPyramid.NLevels() == lPyramid.NLevels()+1);
  for (int level=0; level<gPyramid.NLevels()-1; ++level) {
    //assert(gPyramid.Dim(level) == lPyramid.Dim(level));
    ComputeThisLLevel(gPyramid.GetLevel(level), gPyramid.GetLevel(level+1),
        gPyramid.Dim(level), lPyramid.GetLevel(level));
  }
}

void ConstructPartialGaussianPyramid(ImagePyramid& gPyramid,
                                     const int maxLevel) {
  assert(maxLevel < gPyramid.NLevels());
  for (int level=0; level<maxLevel; ++level) {
    ComputeNextGLevel(gPyramid.GetLevel(level), gPyramid.Dim(level),
        gPyramid.GetLevel(level+1));
  }
}

void ConstructPartialLaplacianPyramid(const ImagePyramid& gPyramid, 
                                      const int maxLevel,
                                      ImagePyramid& lPyramid) {
  assert(maxLevel < lPyramid.NLevels());
  assert(gPyramid.NLevels() == lPyramid.NLevels()+1);
  for (int level=0; level<=maxLevel; ++level) {
    //assert(gPyramid.Dim(level) == lPyramid.Dim(level));
    ComputeThisLLevel(gPyramid.GetLevel(level), gPyramid.GetLevel(level+1),
        gPyramid.Dim(level), lPyramid.GetLevel(level));
  }
}

void CollapseLaplacianPyramid(const ImagePyramid& lPyramid,
                              ImagePyramid& gPyramid, float* const d_image) {
  assert(lPyramid.NLevels() == gPyramid.NLevels()-1);
  for (int level=lPyramid.NLevels()-1; level>=0; --level) {
    //assert(lPyramid.Dim(level) == gPyramid.Dim(level));
    ReconstructThisGLevel(gPyramid.GetLevel(level+1), lPyramid.GetLevel(level),
                          lPyramid.Dim(level), gPyramid.GetLevel(level));
  }
  checkCudaErrors(cudaMemcpy(d_image, gPyramid.GetLevel(0),
      gPyramid.Size(0)*sizeof(float), cudaMemcpyDeviceToDevice));
}

void WriteDeviceImage(const float* d_image, const uint2 dim,
                      const std::string& filename) {
  unsigned int size = dim.x * dim.y;
  unsigned char* d_imageBytes;
  checkCudaErrors(cudaMalloc(&d_imageBytes, size*sizeof(unsigned char)));
  ClampToBytes(d_image, dim, d_imageBytes);

  cv::Mat outImage;
  outImage.create(dim.y, dim.x, CV_8UC1);
  checkCudaErrors(cudaMemcpy(outImage.ptr<unsigned char>(0), d_imageBytes,
      size*sizeof(unsigned char), cudaMemcpyDeviceToHost));
  cudaFree(d_imageBytes);
  cv::imwrite(filename.c_str(), outImage);
}

