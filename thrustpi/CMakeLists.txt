cmake_minimum_required(VERSION 3.0)

project(thrustpi LANGUAGES CXX CUDA)

set(CUDA_NVCC_FLAGS_DEBUG "-g -G")
set(CMAKE_CUDA_FLAGS_DEBUG "-g")
set(CUDA_NVCC_FLAGS "--extended-lambda")

find_package(CUDA 10.2)

add_executable(thrustpi kernel.cu)

target_compile_options(thrustpi PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(thrustpi PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
    
endif()
