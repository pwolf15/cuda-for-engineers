cmake_minimum_required(VERSION 3.0)
project(dist_v1_cuda LANGUAGES CXX CUDA)

set(CUDA_NVCC_FLAGS_DEBUG "-g -G")
set(CMAKE_CUDA_FLAGS_DEBUG "-g")

find_package(CUDA 10.2)

add_executable(dist_v1_cuda kernel.cu)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(dist_v1_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
