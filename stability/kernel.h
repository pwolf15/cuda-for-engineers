#ifndef __KERNEL_H__
#define __KERNEL_H__

struct uchar4;
struct int2;

void kernelLauncher(uchar4* d_out, int w, int h, float p, int s);

#endif