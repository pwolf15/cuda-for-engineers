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

void normNPP(Npp8u *arr, Npp8u* out, int w, int h)
{
  Npp8u *d_in = 0, *d_out = 0;

  cudaMalloc(&d_out, kNumCh*w*h*sizeof(Npp8u));
  cudaMalloc(&d_in, kNumCh*w*h*sizeof(Npp8u));
  cudaMemcpy(d_in, arr, kNumCh*w*h*sizeof(Npp8u),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, out, kNumCh*w*h*sizeof(Npp8u),
             cudaMemcpyHostToDevice);
  const NppiSize oSizeROI = {w, h};
  Npp64f *aNorm = 0;
  cudaMalloc(&aNorm, 3*sizeof(Npp64f));
  Npp8u *pDeviceBuffer = 0;
  int bufferSize;
  nppiNormDiffL2GetBufferHostSize_8u_C3R(oSizeROI, &bufferSize);
  cudaMalloc(&pDeviceBuffer, bufferSize);
  nppiNormDiff_L2_8u_C3R(d_in, kNumCh*w*sizeof(Npp8u), d_out,
    kNumCh*w*sizeof(Npp8u), oSizeROI, aNorm, pDeviceBuffer);
  Npp64f res[3];
  cudaMemcpy(res, aNorm, 3*sizeof(Npp64f), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 3; ++i)
  {
    printf("%f\n", res[i]/(w*h));
  }

  cudaFree(d_in);
  cudaFree(d_out);
}

bool equalHost(Npp8u *host_1, Npp8u *host_2, int numCh, int w, int h)
{
  bool equalCheck = true;
  for (int i = 0; i < w*h*numCh; ++i)
  {
    if (host_1[i] != host_2[i])
    {
      equalCheck = false;
      break;
    }
  }
  return equalCheck;
}

bool equalDev(Npp8u *dev_1, Npp8u *dev_2, int numCh, int w, int h)
{
  Npp8u *host_1 = 0, *host_2 = 0;
  host_1 = (Npp8u*)malloc(numCh*w*h*sizeof(Npp8u));
  host_2 = (Npp8u*)malloc(numCh*w*h*sizeof(Npp8u));

  cudaMemcpy(host_1, dev_1, numCh*w*h*sizeof(Npp8u), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_2, dev_2, numCh*w*h*sizeof(Npp8u), cudaMemcpyDeviceToHost);

  bool equalCheck = true;
  for (int i = 0; i < w*h*numCh; ++i)
  {
    if (host_1[i] != host_2[i])
    {
      equalCheck = false;
      break;
    }
  }

  free(host_1);
  free(host_2);
  return equalCheck;
}

void grayscaleNormNPP(Npp8u *arr, Npp8u* out, int w, int h)
{
  Npp8u *d_in = 0, *d_out = 0, *d_temp_gray1 = 0, *d_temp_gray2 = 0;

  cudaMalloc(&d_out, kNumCh*w*h*sizeof(Npp8u));
  cudaMalloc(&d_in, kNumCh*w*h*sizeof(Npp8u));
  cudaMemcpy(d_in, arr, kNumCh*w*h*sizeof(Npp8u),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, out, kNumCh*w*h*sizeof(Npp8u),
             cudaMemcpyHostToDevice);

  cudaMalloc(&d_temp_gray1, w*h*sizeof(Npp8u));
  cudaMalloc(&d_temp_gray2, w*h*sizeof(Npp8u));

  const NppiSize oSizeROI = {w, h};
  Npp64f *pNormDiff = NULL;
  cudaMalloc(&pNormDiff, sizeof(Npp64f));

  Npp8u* pDeviceBufferGray = NULL;
  int bufferSizeGray;
  nppiNormDiffL2GetBufferHostSize_8u_C1R(oSizeROI, &bufferSizeGray);
  cudaMalloc(&pDeviceBufferGray, bufferSizeGray);

  // Convert to Grayscale
  nppiRGBToGray_8u_C3C1R(d_in, kNumCh*w*sizeof(Npp8u), d_temp_gray1,
    w*sizeof(Npp8u), oSizeROI);
  nppiRGBToGray_8u_C3C1R(d_out, kNumCh*w*sizeof(Npp8u), d_temp_gray2,
    w*sizeof(Npp8u), oSizeROI);

  // grayscale norm diff
  nppiNormDiff_L2_8u_C1R(d_temp_gray1, w*sizeof(Npp8u), d_temp_gray2,
    w*sizeof(Npp8u), oSizeROI, pNormDiff, pDeviceBufferGray);

  Npp64f res = 0;
  cudaMemcpy(&res, pNormDiff, sizeof(Npp64f), cudaMemcpyDeviceToHost);
  printf("%f\n", res/(w*h));
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

  bool testEqual = equalHost(arr, orig, kNumCh, w, h);

  std::cout << "Arr == orig: " << testEqual << std::endl;

  // sharpened
  sharpenNPP(arr, w, h);

  // create copy of sharpened
  Npp8u *sharpened = (Npp8u*)malloc(kNumCh*w*h*sizeof(Npp8u));
  memcpy((void*)sharpened, (void*)arr, kNumCh*w*h*sizeof(Npp8u));

  testEqual = equalHost(sharpened, orig, kNumCh, w, h);

  std::cout << "Sharpened == orig: " << testEqual << std::endl;

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

  // create copy of permuted
  Npp8u *permuted = (Npp8u*)malloc(kNumCh*w*h*sizeof(Npp8u));
  memcpy((void*)permuted, (void*)arr, kNumCh*w*h*sizeof(Npp8u));

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

  // create copy of permuted
  Npp8u *sumImage = (Npp8u*)malloc(kNumCh*w*h*sizeof(Npp8u));
  memcpy((void*)sumImage, (void*)arr, kNumCh*w*h*sizeof(Npp8u));

  image.save_bmp("sum.bmp");

  std::cout << "Norm with itself" << std::endl;
  normNPP(orig, orig, w, h);      // compare with itself

  std::cout << "Norm with sharpened" << std::endl;
  normNPP(sharpened, orig, w, h); // compare wih sharpened

  std::cout << "Norm with color-swapped" << std::endl;
  normNPP(permuted, orig, w, h); // compare with color-swapped

  std::cout << "Grayscale: norm with color-swapped" << std::endl;
  grayscaleNormNPP(permuted, orig, w, h); // compare with grayscale color-swapped

  std::cout << "Norm with sumImage" << std::endl;
  normNPP(sumImage, orig, w, h); // compare with color-swapped

  std::cout << "Grayscale: norm with sumImage" << std::endl;
  grayscaleNormNPP(sumImage, orig, w, h); // compare with grayscale color-swapped

  free(arr);
  return 0;
}
