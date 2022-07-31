cmake_minimum_required(VERSION 3.0)

project(misc_npp LANGUAGES CXX CUDA)

set(CUDA_NVCC_FLAGS_DEBUG "-g -G")
set(CMAKE_CUDA_FLAGS_DEBUG "-g")

find_package(CUDA 10.2)

include_directories(${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/include)

add_executable(misc_npp main.cu)
link_directories(/usr/local/cuda-10.2/targets/aarch64-linux/lib)
target_link_libraries(misc_npp nppif nppidei nppial nppist nppicc)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(misc_npp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
