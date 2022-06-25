#include "aux_functions.h"
#include <stdlib.h>
#include <iostream>

#define N 20000000

int main()
{
  float *in = (float*)calloc(N, sizeof(float));
  float *out = (float*)calloc(N, sizeof(float));
  const float ref = 0.5f;

  for (int i = 0; i < N; ++i)
  {
    in[i] = scale(i, N);
  }

  distanceArray2(out, in, ref, N);

  for (int i = 0; i < 100; ++i)
  {
    std::cout << in[i] << "," << out[i] << std::endl;
  }

  free(in);
  free(out);
  return 0;
}
