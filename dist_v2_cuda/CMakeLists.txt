cmake_minimum_required(VERSION 3.0)

project(dist_v2_cuda)

add_executable(dist_v2 main.cpp kernel.cu)
