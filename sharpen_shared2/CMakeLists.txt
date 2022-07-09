cmake_minimum_required(VERSION 3.0)

project(sharpen_shared2 LANGUAGES CXX CUDA)

set(CUDA_NVCC_FLAGS_DEBUG "-g -G")
set(CMAKE_CUDA_FLAGS_DEBUG "-g")

find_package(CUDA 10.2)

include_directories(${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/include)

add_executable(sharpen_shared2 main.cpp kernel.cu)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(sharpen_shared2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
