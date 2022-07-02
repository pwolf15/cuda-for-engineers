#include <math.h>
#include <stdio.h>
#include <iostream>

#define TPB 32
#define N 64

float scale(int i, int n)
{
  return ((float)i) / (n - 1);
}

__device__
float distance(float x1, float x2)
{
  return sqrt((x2 - x1)*(x2 - x1));
}

__device__
float distance2(float x1, float x2)
{
  return abs(x2 - x1);
}

__global__
void distanceKernel(float *d_out, float *d_in, float ref)
{
  const int i = blockIdx.x*blockDim.x+threadIdx.x;
  const float x = d_in[i];
  d_out[i] = distance(x, ref);
  printf("i = %2d: dist from %f to %f is %f.\n", i, ref, x, d_out[i]);
}

__global__
void distanceKernel2(float *d_out, float *d_in, float ref)
{
  const int i = blockIdx.x*blockDim.x+threadIdx.x;
  const float x = d_in[i];
  d_out[i] = distance2(x, ref);
  printf("i = %2d: dist from %f to %f is %f.\n", i, ref, x, d_out[i]);
}

int main()
{
  float *in = nullptr;
  float *out = nullptr;
  const float ref = 0.5f;

  cudaMallocManaged(&in, N*sizeof(float));
  cudaMallocManaged(&out, N*sizeof(float));

  for (int i = 0; i < N; ++i)
  {
    in[i] = scale(i, N);
  }

  distanceKernel<<<(N+TPB-1)/TPB, TPB>>>(out, in, ref);
  cudaDeviceSynchronize();

  cudaFree(in);
  cudaFree(out);

  return 0;
}
