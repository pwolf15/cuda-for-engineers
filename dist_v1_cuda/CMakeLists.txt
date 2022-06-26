cmake_minimum_required(VERSION 3.0)
project(dist_v1_cuda)
project(dist_v1_cuda LANGUAGES CXX CUDA)

find_package(CUDA 10.2)

add_executable(dist_v1_cuda kernel.cu)
