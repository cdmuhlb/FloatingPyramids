#include "PyramidKernels.h"

#include <cuda_runtime.h>
#include <math_constants.h>
#include <math_functions.h>
#include "utils.h"

/* Kernels */

__global__
void color_to_grey(const uchar4* const inImage,
                   float* const outImage,
                   const uint2 dim) {
  // Get global coordinates
  const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  if ((x >= dim.x) || (y >= dim.y)) return;
  const unsigned int idx = y*dim.x + x;
  const uchar4 rgba = inImage[idx];
  const float channelSum = 0.299f*rgba.x + 0.587f*rgba.y + 0.114f*rgba.z;
  outImage[idx] = channelSum / 255.0f;
}

__global__
void scale_colors(uchar4* const inout,
                  const float* const newI,
                  const uint2 dim) {
  // Get global coordinates
  const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  if ((x >= dim.x) || (y >= dim.y)) return;
  const unsigned int idx = y*dim.x + x;
  uchar4 rgba = inout[idx];

  /*
  float channelSum = (0.299f*rgba.x + 0.587f*rgba.y + 0.114f*rgba.z) / 255.0f;
  //const float delta = (newI[idx] - channelSum) * 255.0f;
  const float ratio = (newI[idx] + 1.0f) / (channelSum + 1.0f);

  //float newR = rgba.x + 0.299f*delta;
  //float newR = rgba.x * ratio;
  float newR = (rgba.x + 255.0f) * ratio - 255.0f;
  if (newR < 0.0f) newR = 0.0f;
  if (newR > 255.0f) newR = 255.0f;
  rgba.x = newR;

  //float newG = rgba.y + 0.587f*delta;
  //float newG = rgba.y * ratio;
  float newG = (rgba.y + 255.0f) * ratio - 255.0f;
  if (newG < 0.0f) newG = 0.0f;
  if (newG > 255.0f) newG = 255.0f;
  rgba.y = newG;

  //float newB = rgba.z + 0.114f*delta;
  //float newB = rgba.z * ratio;
  float newB = (rgba.z + 255.0f) * ratio - 255.0f;
  if (newB < 0.0f) newB = 0.0f;
  if (newB > 255.0f) newB = 255.0f;
  rgba.z = newB;
  */

  //const float oldY = (0.299f*rgba.x + 0.587f*rgba.y + 0.114f*rgba.z) / 255.0f;
  const float newY = newI[idx];
  const float cb = 0.5f + (-0.168736f*rgba.x - 0.331264f*rgba.y + 0.5f*rgba.z) /
                          255.0f;
  const float cr = 0.5f + (0.5f*rgba.x - 0.418688f*rgba.y - 0.081312f*rgba.z) /
                          255.0f;

  rgba.x = (newY + 1.402f*(cr - 0.5f))*255.0f;
  rgba.y = (newY - 0.34414*(cb - 0.5f) - 0.71414f*(cr - 0.5f))*255.0f;
  rgba.z = (newY + 1.772*(cb - 0.5f))*255.0f;

  // Swap colors
  unsigned char swp = rgba.x;
  rgba.x = rgba.z;
  rgba.z = swp;

  inout[idx] = rgba;
}

__global__
void downsample(const float* const inImage,
                float* const outImage,
                const uint2 dim) {
  // Get global coordinates
  const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int xl = 2*x;
  const unsigned int yl = 2*y;
  const unsigned int xr = xl + 1;
  const unsigned int yr = yl + 1;
  const unsigned int nx = dim.x;
  if ((xr >= nx) || (yr >= dim.y)) return;
  outImage[y*nx/2 + x] = (inImage[yl*nx + xl] + inImage[yl*nx + xr] +
                          inImage[yr*nx + xl] + inImage[yr*nx + xr]) * 0.25f;
}

__global__
void upsample(const float* const inImage,
              float* const outImage,
              const uint2 dim) {
  // Get global coordinates
  const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int nx = dim.x;
  if ((x >= nx) || (y >= dim.y)) return;

  // Cosine^2 weights
  //const float w1 = 0.7220079f;
  //const float w3 = 0.1043749f;

  // Sinc weights
  const float w1 = 0.8067005f;
  const float w3 = 0.2462075f;

  const float vC = inImage[y*nx + x];
  const float wC = w1;

  float vN;
  float wN;
  if (y > 0) {
    vN = inImage[(y-1)*nx + x];
    wN = w3;
  } else {
    vN = 0.0f;
    wN = 0.0f;
  }

  float vE;
  float wE;
  if (x+1 < nx) {
    vE = inImage[y*nx + x+1];
    wE = w3;
  } else {
    vE = 0.0f;
    wE = 0.0f;
  }

  float vS;
  float wS;
  if (y+1 < dim.y) {
    vS = inImage[(y+1)*nx + x];
    wS = w3;
  } else {
    vS = 0.0f;
    wS = 0.0f;
  }

  float vW;
  float wW;
  if (x > 0) {
    vW = inImage[y*nx + x-1];
    wW = w3;
  } else {
    vW = 0.0f;
    wW = 0.0f;
  }

  const unsigned int xl = 2*x;
  const unsigned int yl = 2*y;
  const unsigned int xr = xl + 1;
  const unsigned int yr = yl + 1;
  const unsigned int nx2 = 2*nx;
  outImage[yl*nx2 + xl] = (vC*wC + vW*wW + vN*wN) / (wC + wW + wN);
  outImage[yl*nx2 + xr] = (vC*wC + vE*wE + vN*wN) / (wC + wE + wN);
  outImage[yr*nx2 + xl] = (vC*wC + vW*wW + vS*wS) / (wC + wW + wS);
  outImage[yr*nx2 + xr] = (vC*wC + vE*wE + vS*wS) / (wC + wE + wS);
}

__global__
void residual(const float* const orig,
              float* const otherAndResid,
              const uint2 dim) {
  const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  if ((x >= dim.x) || (y >= dim.y)) return;
  const unsigned int idx = y*dim.x + x;
  otherAndResid[idx] = orig[idx] - otherAndResid[idx];
}

__global__
void add_resid(const float* const resid,
               float* const otherAndSum,
               const uint2 dim) {
  const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  if ((x >= dim.x) || (y >= dim.y)) return;
  const unsigned int idx = y*dim.x + x;
  otherAndSum[idx] += resid[idx];
}

__global__
void apply_filter(const float* const in,
                  float* const out,
                  const uint2 dim,
                  const float ymax) {
  // Get global coordinates
  const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  if ((x >= dim.x) || (y >= dim.y)) return;
  const unsigned int idx = y*dim.x + x;

  const float v = in[idx];

  const float tau = 0.860333589f;       // tau * tan(tau) = 1
  const float mu  = 0.652184624f;       // mu = cos(tau)
  const float a = (ymax > mu) ? (tau / acosf(ymax)) : (ymax / mu);
  out[idx] = (fabsf(v) > a) ? copysignf(ymax, v) : (v*cosf(v*tau/a));

  //const float xtrans = 0.5;
  //out[idx] = v * ((vnorm > xtrans) ? 1.0 :
  //    sinf(fabsf(v)*CUDART_PIO2_F/xtrans));
}

__global__
void apply_fast_ll_filter(const float* const gThis,
                          float* const lThis,
                          const uint2 dim,
                          float** lsByGamma,
                          const int nGammas) {
  const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  if ((x >= dim.x) || (y >= dim.y)) return;
  const unsigned int idx = y*dim.x + x;

  const float g0 = gThis[idx];
  int gammaNum = g0 * (nGammas - 1);
  if (gammaNum < 0) gammaNum = 0;
  if (gammaNum >= nGammas - 1) {
    lThis[idx] = lsByGamma[nGammas-1][idx];
  } else {
    const float a = (g0 - gammaNum/(nGammas - 1.0f)) * (nGammas - 1.0f);
    lThis[idx] = lsByGamma[gammaNum][idx]*(1.0f - a) +
                 lsByGamma[gammaNum+1][idx]*a;
  }
}

__global__
void remap(const float* const in,
           float* const out,
           const uint2 dim,
           const float g0,
           const float alpha,
           const float beta,
           const float sigma) {
  const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  if ((x >= dim.x) || (y >= dim.y)) return;
  const unsigned int idx = y*dim.x + x;

  const float v = in[idx];
  const float delta = fabsf(v - g0)/sigma;
  if (delta < 1.0f) {
    const float t = 0.25f;
    const float tau = (delta < t) ? 0.0f :
        ((delta > 2.0f*t) ? 1.0f : ((delta - t)/t));
    //const float tau = 1.0f;
    const float fd = tau*powf(delta, alpha) + (1.0f - tau)*delta;
    out[idx] = g0 + copysignf(sigma*fd, v - g0);
  } else {
    out[idx] = g0 + copysignf(beta*(fabsf(v - g0) - sigma) + sigma, v - g0);
  }
}

__global__
void shift_and_stretch(float* const inout,
                       const uint2 dim,
                       const float stretch,
                       const float shift) {
  const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  if ((x >= dim.x) || (y >= dim.y)) return;
  const unsigned int idx = y*dim.x + x;

  inout[idx] = stretch*(inout[idx] - 0.5f) + 0.5f + shift;
}

__global__
void weighted_average(float* const a,
                      const float* const b,
                      const uint2 dim,
                      const float wa,
                      const float wb) {
  const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  if ((x >= dim.x) || (y >= dim.y)) return;
  const unsigned int idx = y*dim.x + x;

  a[idx] = wa*a[idx] + wb*b[idx];
}

__global__
void clamp_to_bytes(const float* const in,
                    unsigned char* const out,
                    const uint2 dim) {
  const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  if ((x >= dim.x) || (y >= dim.y)) return;
  const unsigned int idx = y*dim.x + x;

  const float v = 255.0f*in[idx];
  out[idx] = (v < 0.0f) ? 0 : ((v > 255.0f) ? 255 : v);
}


/* Host Functions */

void ConvertToGreyscale(const uchar4* const in,
                        const uint2 dim,
                        float* const out) {
  const dim3 blockSize(16, 16);
  const unsigned int gridx = (dim.x - 1)/blockSize.x + 1;
  const unsigned int gridy = (dim.y - 1)/blockSize.y + 1;
  const dim3 gridSize(gridx, gridy);
  color_to_grey<<<gridSize, blockSize>>>(in, out, dim);
}

void ScaleColors(uchar4* const inout,
                 const float* const newI,
                 const uint2 dim) {
  const dim3 blockSize(16, 16);
  const unsigned int gridx = (dim.x - 1)/blockSize.x + 1;
  const unsigned int gridy = (dim.y - 1)/blockSize.y + 1;
  const dim3 gridSize(gridx, gridy);
  scale_colors<<<gridSize, blockSize>>>(inout, newI, dim);
}

void ComputeNextGLevel(const float* const gThis,
                       const uint2 dim,
                       float* const gNext) {
  const dim3 blockSize(16, 16);
  const unsigned int gridx = (dim.x/2 - 1)/blockSize.x + 1;
  const unsigned int gridy = (dim.y/2 - 1)/blockSize.y + 1;
  const dim3 gridSize(gridx, gridy);
  downsample<<<gridSize, blockSize>>>(gThis, gNext, dim);
}

void ComputeThisLLevel(const float* const gThis,
                       const float* const gNext,
                       const uint2 dim,
                       float* const lThis) {
  const dim3 blockSize(16, 16);
  const uint2 nextDim = {dim.x/2, dim.y/2};
  const unsigned int gridx = (nextDim.x - 1)/blockSize.x + 1;
  const unsigned int gridy = (nextDim.y - 1)/blockSize.y + 1;
  const dim3 gridSize(gridx, gridy);
  upsample<<<gridSize, blockSize>>>(gNext, lThis, nextDim);

  const int gridx2 = (dim.x - 1)/blockSize.x + 1;
  const int gridy2 = (dim.x - 1)/blockSize.y + 1;
  const dim3 gridSize2(gridx2, gridy2);
  residual<<<gridSize2, blockSize>>>(gThis, lThis, dim);
}

void ReconstructThisGLevel(const float* const gNext,
                           const float* const lThis,
                           const uint2 dim,
                           float* const gThis) {
  const dim3 blockSize(16, 16);
  const uint2 nextDim = {dim.x/2, dim.y/2};
  const unsigned int gridx = (nextDim.x - 1)/blockSize.x + 1;
  const unsigned int gridy = (nextDim.y - 1)/blockSize.y + 1;
  const dim3 gridSize(gridx, gridy);
  upsample<<<gridSize, blockSize>>>(gNext, gThis, nextDim);

  const int gridx2 = (dim.x - 1)/blockSize.x + 1;
  const int gridy2 = (dim.x - 1)/blockSize.y + 1;
  const dim3 gridSize2(gridx2, gridy2);
  add_resid<<<gridSize2, blockSize>>>(lThis, gThis, dim);
}

void FilterThisLLevel(const float* const lThisIn,
                      const uint2 dim,
                      const float ymax,
                      float* const lThisOut) {
  const dim3 blockSize(16, 16);
  const unsigned int gridx = (dim.x - 1)/blockSize.x + 1;
  const unsigned int gridy = (dim.y - 1)/blockSize.y + 1;
  const dim3 gridSize(gridx, gridy);
  apply_filter<<<gridSize, blockSize>>>(lThisIn, lThisOut, dim, ymax);
}

void ApplyLLFilter(const float* const gThis,
                   const uint2 dim,
                   float** lsByGamma,
                   const int nGammas,
                   float* const lThis) {
  const dim3 blockSize(16, 16);
  const unsigned int gridx = (dim.x - 1)/blockSize.x + 1;
  const unsigned int gridy = (dim.y - 1)/blockSize.y + 1;
  const dim3 gridSize(gridx, gridy);
  apply_fast_ll_filter<<<gridSize, blockSize>>>(gThis, lThis, dim, lsByGamma, nGammas);
}

void ShiftAndStretch(float* const inout,
                     const uint2 dim,
                     const float stretch,
                     const float shift) {
  const dim3 blockSize(16, 16);
  const unsigned int gridx = (dim.x - 1)/blockSize.x + 1;
  const unsigned int gridy = (dim.y - 1)/blockSize.y + 1;
  const dim3 gridSize(gridx, gridy);
  shift_and_stretch<<<gridSize, blockSize>>>(inout, dim, stretch, shift);
}

void WeightedAverage(float* const a,
                     const float* const b,
                     const uint2 dim,
                     const float wa,
                     const float wb) {
  const dim3 blockSize(16, 16);
  const unsigned int gridx = (dim.x - 1)/blockSize.x + 1;
  const unsigned int gridy = (dim.y - 1)/blockSize.y + 1;
  const dim3 gridSize(gridx, gridy);
  weighted_average<<<gridSize, blockSize>>>(a, b, dim, wa, wb);
}

void RemapImage(const float* const in,
                const uint2 dim,
                const float g0,
                const float alpha,
                const float beta, 
                const float sigma, 
                float* const out) {
  const dim3 blockSize(16, 16);
  const unsigned int gridx = (dim.x - 1)/blockSize.x + 1;
  const unsigned int gridy = (dim.y - 1)/blockSize.y + 1;
  const dim3 gridSize(gridx, gridy);
  remap<<<gridSize, blockSize>>>(in, out, dim, g0, alpha, beta, sigma);
}

void ClampToBytes(const float* const in,
                  const uint2 dim, 
                  unsigned char* const out) {
  const dim3 blockSize(16, 16);
  const unsigned int gridx = (dim.x - 1)/blockSize.x + 1;
  const unsigned int gridy = (dim.y - 1)/blockSize.y + 1;
  const dim3 gridSize(gridx, gridy);
  clamp_to_bytes<<<gridSize, blockSize>>>(in, out, dim);
}

