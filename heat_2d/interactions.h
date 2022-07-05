#ifndef __INTERACTIONS_H__
#define __INTERACTIONS_H__
#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glew.h>
#include <GL/freeglut.h>
#endif
#define MAX(x,y) (((x) > (y)) ? (x) : (y))
#define W 640
#define H 640
#define DT 1.f

float *d_temp = 0;
int iterationCount = 0;
BC bc = { W/2, H/2, W/10.f, 150, 212.f, 70.f, 0.0f};

void keyboard(unsigned char key, int x, int y)
{
  if (key == 'S') bc.t_s += DT;
  if (key == 's') bc.t_s -= DT;
  if (key == 'A') bc.t_a += DT;
  if (key == 'a') bc.t_a -= DT;
  if (key == 'G') bc.t_g += DT;
  if (key == 'g') bc.t_g -= DT;
  if (key == 'R') bc.rad += DT;
  if (key == 'r') bc.rad = MAX(0.f, bc.rad-DT);
  if (key == 'C') ++bc.chamfer;
  if (key == 'c') --bc.chamfer;
  if (key == 'z') resetTemperature(d_temp, W, H, bc);
  if (key == 27) exit(0);
  glutPostRedisplay();
}

void mouse(int button, int state, int x, int y)
{
  bc.x = x, bc.y = y;
  glutPostRedisplay();
}

void idle()
{
  ++iterationCount;
  glutPostRedisplay();
}

void printInstructions() 
{
  printf("Temperature Visualizer:\n");
  printf("Relocate source with mouse click\n");
  printf("Change source temperatiree (-/+) : s/S\n");
  printf("Change air temperature     (-/+) : a/A\n");
  printf("Change ground temperature  (-/+) : g/G\n");
  printf("Change pipe radius         (-/+) : r/R\n");
  printf("Change chamfer             (-/+) : c/C\n");
  printf("Reset to air temperature   (-/+) : z/Z\n");
  printf("Exit                             : Esc\n");
}

#endif