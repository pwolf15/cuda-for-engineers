cmake_minimum_required(VERSION 3.0)

project(dd_1d_global LANGUAGES CXX CUDA)

set(CUDA_NVCC_FLAGS_DEBUG "-g -G")
set(CMAKE_CUDA_FLAGS_DEBUG "-g")

find_package(CUDA 10.2)

add_executable(dd_1d_global main.cpp kernel.cu)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(dd_1d_global PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
