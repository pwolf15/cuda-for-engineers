cmake_minimum_required(VERSION 3.0)

project(dist_2d_image LANGUAGES CXX CUDA)

set(CUDA_NVCC_FLAGS_DEBUG "-g -G")
set(CMAKE_CUDA_FLAGS_DEBUG "-g")

find_package(CUDA 10.2)

add_executable(dist_2d_image kernel.cu)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(dist_2d_image PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
