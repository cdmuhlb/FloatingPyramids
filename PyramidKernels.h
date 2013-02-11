#ifndef PYRAMIDKERNELS_H_
#define PYRAMIDKERNELS_H_

#include <vector_types.h>

void ConvertToGreyscale(const uchar4* in, uint2 dim, float* out);

void ComputeNextGLevel(const float* gThis, uint2 dim, float* gNext);

void ComputeThisLLevel(const float* gThis, const float* gNext, uint2 dim,
                       float* lThis);

void ReconstructThisGLevel(const float* gNext, const float* lThis, uint2 dim,
                           float* gThis);

void FilterThisLLevel(const float* lThisIn, uint2 dim, float ymax,
                      float* lThisOut);

void ApplyLLFilter(const float* gThis, uint2 dim, float** lsByGamma,
                   int nGammas, float* lThis);

void ShiftAndStretch(float* inout, uint2 dim, float stretch, float shift);

void WeightedAverage(float* a, const float* b, uint2 dim, float wa, float wb);

void RemapImage(const float* in, uint2 dim, float g0, float alpha, float beta,
                float sigma, float* out);

void ClampToBytes(const float* in, uint2 dim, unsigned char* out);

#endif  // PYRAMIDKERNELS_H_

