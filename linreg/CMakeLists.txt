cmake_minimum_required(VERSION 3.0)
project(linreg LANGUAGES CXX CUDA)

set(CUDA_NVCC_FLAGS_DEBUG "-g -G")
set(CMAKE_CUDA_FLAGS_DEBUG "-g")

find_package(CUDA 10.2)

add_executable(linreg kernel.cu)
target_link_libraries(linreg cublas cusolver)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(linreg PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
