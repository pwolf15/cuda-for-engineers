#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/generate.h>
#include <thrust/transform.h>
#include <math.h>
#include <stdio.h>
#define N (1 << 20)

using namespace thrust::placeholders;

int main()
{
  thrust::host_vector<float> hvec_x(N);
  thrust::host_vector<float> hvec_y(N);
  thrust::generate(hvec_x.begin(), hvec_x.end(), rand);
  thrust::generate(hvec_y.begin(), hvec_y.end(), rand);
  thrust::device_vector<float> dvec_x = hvec_x;
  thrust::device_vector<float> dvec_y = hvec_y;
  thrust::transform(dvec_x.begin(), dvec_x.end(), dvec_x.begin(),
    _1 / RAND_MAX);
  thrust::transform(dvec_y.begin(), dvec_y.end(), dvec_y.begin(),
    _1 / RAND_MAX);
  thrust::device_vector<float> dvec_inCircle(N);
  thrust::transform(dvec_x.begin(), dvec_x.end(), dvec_y.begin(),
    dvec_inCircle.begin(), (_1*_1 + _2*_2)<1);
  float pi =
    thrust::reduce(dvec_inCircle.begin(), dvec_inCircle.end())*4.f/N;
  printf("pi = %f\n", pi);

  return 0;
}