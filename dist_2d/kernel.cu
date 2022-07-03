#include <math.h>
#include <stdio.h>
#define W 500
#define H 500
#define TX 500
#define TY 1

__global__
void distanceKernel(float *d_out, int w, int h, float2 pos)
{
  const int c = blockIdx.x*blockDim.x + threadIdx.x;
  const int r = blockIdx.y*blockDim.y + threadIdx.y;
  const int i = r*w + c;
  if ((c >= w) || (r >= h)) return;

  d_out[i] = sqrtf(((c-pos.x)*(c-pos.x))+((r-pos.y)*(r-pos.y)));
  printf("%f\n", d_out[i]);
}

int main()
{
  float *d_out = nullptr;
  float *out = (float*)calloc(W*H, sizeof(float));

  cudaMalloc(&d_out, W*H*sizeof(float));

  const float2 pos = {0.0f,0.0f};
  const dim3 blockSize{TX, TY};
  const int bx = 1;
  const int by = 1;
  const dim3 gridSize{bx, by};
  distanceKernel<<<gridSize, blockSize>>>(d_out, W, H, pos);

  cudaMemcpy(out, d_out, W*H*sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_out);
  free(out);
  return 0;
}