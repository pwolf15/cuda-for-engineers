#define cimg_display 0
#include "CImg.h"
#include <cuda_runtime.h>
#include <npp.h>
#include <stdlib.h>
#define kNumCh 3
#include <iostream>

void sharpenNPP(Npp8u *arr, int w, int h)
{
  Npp8u *d_in = 0, *d_out = 0;
  Npp32f *d_filter = 0;
  const Npp32f filter[9] =
  {
    -1.0, -1.0, -1.0,
    -1.0,  9.0, -1.0,
    -1.0, -1.0, -1.0
  };
  cudaMalloc(&d_out, kNumCh*w*h*sizeof(Npp8u));
  cudaMalloc(&d_in, kNumCh*w*h*sizeof(Npp8u));
  cudaMalloc(&d_filter, 9*sizeof(Npp32f));
  cudaMemcpy(d_in, arr, kNumCh*w*h*sizeof(Npp8u),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_filter, filter, 9*sizeof(Npp32f),
             cudaMemcpyHostToDevice);
  const NppiSize oKernelSize = { 3, 3 };
  const NppiPoint oAnchor = {1, 1};
  const NppiSize oSrcSize = {w, h};
  const NppiPoint oSrcOffset = {0, 0};
  const NppiSize oSizeROI = {w, h};

  nppiFilterBorder32f_8u_C3R(d_in, kNumCh*w*sizeof(Npp8u), oSrcSize,
    oSrcOffset, d_out, kNumCh*w*sizeof(Npp8u), oSizeROI, d_filter,
    oKernelSize, oAnchor, NPP_BORDER_REPLICATE);
  
  cudaMemcpy(arr, d_out, kNumCh*w*h*sizeof(Npp8u),
             cudaMemcpyDeviceToHost);
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_filter);
}

void permuteNPP(Npp8u *arr, int w, int h)
{
  Npp8u *d_in = 0, *d_out = 0;

  cudaMalloc(&d_out, kNumCh*w*h*sizeof(Npp8u));
  cudaMalloc(&d_in, kNumCh*w*h*sizeof(Npp8u));
  cudaMemcpy(d_in, arr, kNumCh*w*h*sizeof(Npp8u),
             cudaMemcpyHostToDevice);
  const NppiSize oSizeROI = {w, h};
  const int aDstOrder[3] = {1,2,0};

  nppiSwapChannels_8u_C3R(d_in, kNumCh*w*sizeof(Npp8u), d_out,
    kNumCh*w*sizeof(Npp8u), oSizeROI, aDstOrder);
  
  cudaMemcpy(arr, d_out, kNumCh*w*h*sizeof(Npp8u),
             cudaMemcpyDeviceToHost);
  cudaFree(d_in);
  cudaFree(d_out);
}

void sumNPP(Npp8u *arr, Npp8u* out, int w, int h)
{
  Npp8u *d_in = 0, *d_out = 0;

  cudaMalloc(&d_out, kNumCh*w*h*sizeof(Npp8u));
  cudaMalloc(&d_in, kNumCh*w*h*sizeof(Npp8u));
  cudaMemcpy(d_in, arr, kNumCh*w*h*sizeof(Npp8u),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, out, kNumCh*w*h*sizeof(Npp8u),
             cudaMemcpyHostToDevice);
  const NppiSize oSizeROI = {w, h};

  nppiAdd_8u_C3RSfs(d_in, kNumCh*w*sizeof(Npp8u), d_out,
    kNumCh*w*sizeof(Npp8u), d_out, kNumCh*w*sizeof(Npp8u), oSizeROI, 1);
  
  cudaMemcpy(arr, d_out, kNumCh*w*h*sizeof(Npp8u),
             cudaMemcpyDeviceToHost);
  cudaFree(d_in);
  cudaFree(d_out);
}


int main()
{
  cimg_library::CImg<unsigned char> image("/home/pwolf/dev/cuda_for_engineers/misc_npp/build/Tricoloring.png");
  const int w = image.width();
  const int h = image.height();
  Npp8u *arr = (Npp8u*)malloc(kNumCh*w*h*sizeof(Npp8u));

  for (int r = 0; r < h; ++r)
  {
    for (int c = 0; c < w; ++c)
    {
      for (int ch = 0; ch < kNumCh; ++ch)
      {
        arr[kNumCh*(r*w + c) + ch] = image(c, r, ch);
      }
    }
  }

  // create copy of original
  Npp8u *orig = (Npp8u*)malloc(kNumCh*w*h*sizeof(Npp8u));
  memcpy((void*)orig, (void*)arr, kNumCh*w*h*sizeof(Npp8u));

  // sharpened
  sharpenNPP(arr, w, h);

  for (int r = 0; r < h; ++r)
  {
    for (int c = 0; c < w; ++c)
    {
      for (int ch = 0; ch < kNumCh; ++ch)
      {
        image(c, r, ch) = arr[kNumCh*(r*w + c) + ch];
      }
    }
  }

  image.save_bmp("out.bmp");

  // permuted
  permuteNPP(arr, w, h);

  for (int r = 0; r < h; ++r)
  {
    for (int c = 0; c < w; ++c)
    {
      for (int ch = 0; ch < kNumCh; ++ch)
      {
        image(c, r, ch) = arr[kNumCh*(r*w + c) + ch];
      }
    }
  }

  // sum original and permuted
  sumNPP(arr, orig, w, h);

  for (int r = 0; r < h; ++r)
  {
    for (int c = 0; c < w; ++c)
    {
      for (int ch = 0; ch < kNumCh; ++ch)
      {
        image(c, r, ch) = arr[kNumCh*(r*w + c) + ch];
      }
    }
  }

  image.save_bmp("sum.bmp");

  free(arr);
  return 0;
}
