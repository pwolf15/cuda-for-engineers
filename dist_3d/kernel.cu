#include <stdio.h>
#define W 32
#define H 32
#define D 32
#define TX 8
#define TY 8
#define TZ 8

int divUp(int a, int b) { return (a + b - 1)/b; }

__device__
float distance(int c, int r, int s, float3 pos)
{
  return sqrtf((c - pos.x)*(c - pos.x) + (r - pos.y)*(r - pos.y) +
               (s - pos.z)*(s - pos.z));
}

__global__
void distanceKernel(float *d_out, int w, int h, int d, float3 pos)
{
  const int c = blockIdx.x*blockDim.x + threadIdx.x;
  const int r = blockIdx.y*blockDim.y + threadIdx.y;
  const int s = blockIdx.z*blockDim.z + threadIdx.z;
  const int i = s*w*h + r*w + c;
  if ((c >= w) || (r >= h) || (s >= d)) return;
  d_out[i] = distance(c, r, s, pos);
  printf("%f\n", d_out[i]);
}

int main()
{
  float *out = (float*)calloc(W*H*D, sizeof(float));
  float *d_out = nullptr;
  cudaMalloc(&d_out, W*H*D*sizeof(float));
  const float3 pos = { 0.0f, 0.0f, 0.0f };
  const dim3 blockSize(TX, TY, TZ);
  const dim3 gridSize(divUp(W, TX), divUp(H, TY), divUp(D, TZ));
  distanceKernel<<<gridSize, blockSize>>>(d_out, W, H, D, pos);
  cudaMemcpy(out, d_out, W*H*D*sizeof(float), cudaMemcpyDeviceToHost);
  free(out);
  return 0;
}