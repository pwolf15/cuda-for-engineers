#include "kernel.h"
#define TX 128
#define TY 8
#define RAD 1

int divUp(int a, int b) { return (a + b - 1) / b; }

__device__
unsigned char clip(int n) { return n > 255 ? 255 : (n < 0 ? 0 : n); }

__device__
int idxClip(int idx, int idxMax)
{
  return idx > (idxMax - 1) ? (idxMax - 1) : (idx < 0 ? 0 : idx);
}

__device__
int flatten(int col, int row, int width, int height)
{
  return idxClip(col, width) + idxClip(row, height)*width;
}

__global__
void sharpenKernel(uchar4* d_out, uchar4* d_in, const float *d_filter, int w, int h)
{
  extern __shared__ uchar4 s_in[];
  const int c = threadIdx.x+blockDim.x*blockIdx.x;
  const int r = threadIdx.y+blockDim.y*blockIdx.y;
  if ((c >= w) || (r >= h)) return;
  const int i = flatten(c, r, w, h);
  const int s_c = threadIdx.x + RAD;
  const int s_r = threadIdx.y + RAD;
  const int s_w = blockDim.x + 2*RAD;
  const int s_h = blockDim.y + 2*RAD;
  const int s_i = flatten(s_c, s_r, s_w, s_h);
  const int fltSz = 2*RAD + 1;

  // Regular cells
  s_in[s_i] = d_in[i];

  // Halo cells

  // corner halo values
  if (threadIdx.x < RAD && threadIdx.y < RAD) 
  {
    s_in[flatten(s_c - RAD, s_r - RAD, s_w, s_h)] =
      d_in[flatten(c - RAD, r - RAD, w, h)];
    s_in[flatten(s_c + blockDim.x, s_r - RAD, s_w, s_h)] = 
      d_in[flatten(c + blockDim.x, r - RAD, w, h)];
    s_in[flatten(s_c - RAD, s_r + blockDim.y, s_w, s_h)] =
      d_in[flatten(c - RAD, r + blockDim.y, w, h)];
    s_in[flatten(s_c + blockDim.y, s_r + blockDim.y, s_w, s_h)] =
      d_in[flatten(c + blockDim.y, r + blockDim.y, w, h)];
  }

  // edge halo values
  if (threadIdx.x < RAD)
  {
    s_in[flatten(s_c - RAD, s_r, s_w, s_h)] = 
      d_in[flatten(c - RAD, r, w, h)];
    s_in[flatten(s_c + blockDim.x, s_r, s_w, s_h)] =
      d_in[flatten(c + blockDim.x, r, w, h)];
  }
  if (threadIdx.y < RAD)
  {
    s_in[flatten(s_c, s_r - RAD, s_w, s_h)] = 
      d_in[flatten(c, r - RAD, w, h)];
    s_in[flatten(s_c, s_r + blockDim.y, s_w, s_h)] =
      d_in[flatten(c, r + blockDim.y, w, h)];
  }
  __syncthreads();

  float rgb[3] = {0.f, 0.f, 0.f};

  for (int rd = -RAD; rd <= RAD; ++rd)
  {
    for (int cd = -RAD; cd <= RAD; ++cd)
    {
      const int s_imgIdx = flatten(s_c + cd, s_r + rd, s_w, s_h);
      const int fltIdx = flatten(RAD + cd, RAD + rd, fltSz, fltSz);
      uchar4 color = s_in[s_imgIdx];
      float weight = d_filter[fltIdx];
      rgb[0] += weight*color.x;
      rgb[1] += weight*color.y;
      rgb[2] += weight*color.z;
    }
  }
  d_out[i].x = clip(rgb[0]);
  d_out[i].y = clip(rgb[1]);
  d_out[i].z = clip(rgb[2]);
}

void sharpenParallel(uchar4* arr, int w, int h)
{
  const int fltSz = 2 * RAD + 1;
  const float filter[9] = 
  {
    -1.0, -1.0, -1.0,
    -1.0,  9.0, -1.0,
    -1.0, -1.0, -1.0
  };

  uchar4 *d_in = 0, *d_out = 0;
  float *d_filter = 0;

  cudaMalloc(&d_in, w*h*sizeof(uchar4));
  cudaMemcpy(d_in, arr, w*h*sizeof(uchar4), cudaMemcpyHostToDevice);

  cudaMalloc(&d_out, w*h*sizeof(uchar4));

  cudaMalloc(&d_filter, fltSz*fltSz*sizeof(float));
  cudaMemcpy(d_filter, filter, fltSz*fltSz*sizeof(float), cudaMemcpyHostToDevice);

  const dim3 blockSize(TX, TY);
  const dim3 gridSize(divUp(w, blockSize.x), divUp(h, blockSize.y));
  const size_t smSz = (TX+2*RAD)*(TY+2*RAD)*sizeof(uchar4);

  sharpenKernel<<<gridSize, blockSize, smSz>>>(d_out, d_in, d_filter, w, h);

  cudaMemcpy(arr, d_out, w*h*sizeof(uchar4), cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_filter);
}