cmake_minimum_required(VERSION 3.0)

project(stability LANGUAGES CXX CUDA)

set(CUDA_NVCC_FLAGS_DEBUG "-g -G")
set(CMAKE_CUDA_FLAGS_DEBUG "-g")

find_package(GLUT REQUIRED)
find_package(CUDA 10.2)

include_directories(${GLUT_INCLUDE_DIRS})
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/include)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)

add_executable(stability main.cpp kernel.cu)

target_link_libraries(stability ${GLUT_LIBRARIES} GL GLU ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/lib/linux/aarch64/libGLEW.a)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(stability PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
