#ifndef __INTERACTIONS_H__
#define __INTERACTIONS_H__
#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include "windows.h"
#endif
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glew.h>
#include <GL/freeglut.h>
#endif
#include <vector_types.h>
#define W 600
#define H 600
#define DELTA 5
#define NX 128
#define NY 128
#define NZ 128

int id = 1; // 0 = sphere, 1 = torus, 2 = block
int method = 2; // 0 = volumeRender, 1 = slice, 2 = raycast
const int3 volumeSize = { NX, NY, NZ };
const float4 params = { NX / 4.f, NY / 6.f, NZ / 16.f, 1.f };
float *d_vol;
float zs = NZ;
float dist = 0.f, theta = 0.f, threshold = 0.f;

void mymenu(int value)
{
  switch(value) 
  {
    case 0: return;
    case 1: id = 0; break; // sphere
    case 2: id = 1; break; // torus
    case 3: id = 2; break; // block
  }
  volumeKernelLauncher(d_vol, volumeSize, id, params);
  glutPostRedisplay();
}

void createMenu()
{
  glutCreateMenu(mymenu);
  glutAddMenuEntry("Object Selector", 0);
  glutAddMenuEntry("Sphere", 1);
  glutAddMenuEntry("Torus", 2);
  glutAddMenuEntry("Block", 3);
  glutAttachMenu(GLUT_RIGHT_BUTTON);
}

void keyboard(unsigned char key, int x, int y)
{
  if (key == '+') zs -= DELTA; // move source closer
  if (key == '-') zs += DELTA; // move source farther
  if (key == 'd') --dist; // decrease slice distance
  if (key == 'D') ++dist; // increase slice distance
  if (key == 'z') zs = Nz, theta = 0.f, dist = 0.f; // reset values
  if (key == 'v') method = 0; // volume rendering
  if (key == 'f') method = 1; // slicing
  if (key == 'r') method = 2; // racyast
  if (key == 27) exit(0);
  glutPostRedisplay();
}

void handleSpecialKeypress(int key, int x, int y)
{
  if (key == GLUT_KEY_LEFT) theta -= 0.1f;
  if (key == GLUT_KEY_RIGHT) theta += 0.1f;
  if (key == GLUT_KEY_UP) threshold += 0.1f;
  if (key == GLUT_KEY_DOWN) threshold -= 0.1f;
  glutPostRedisplay();
}

void printInstructions() 
{
  printf("flashlight interactions\n");
  printf("a: toggle mouse tracking mode\n");
  printf("arrow keys: move ref location\n");
  printf("esc: close graphics window\n");
}

#endif