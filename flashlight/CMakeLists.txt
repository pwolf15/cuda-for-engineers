cmake_minimum_required(VERSION 3.0)

project(flashlight LANGUAGES CXX CUDA)

set(CUDA_NVCC_FLAGS_DEBUG "-g -G")
set(CMAKE_CUDA_FLAGS_DEBUG "-g")

find_package(GLUT REQUIRED)
find_package(CUDA 10.2)

message(${GLUT_INCLUDE_DIR})
include_directories(${GLUT_INCLUDE_DIRS})
include_directories(/usr/local/cuda-10.2/targets/aarch64-linux/include)
include_directories(/usr/local/cuda-10.2/samples/common/inc)
add_executable(flashlight main.cpp kernel.cu)
message(${GLUT_LIBRARIES})
link_directories(/usr/local/cuda-10.2/samples/common/lib/linux/aarch64/)
target_link_libraries(flashlight ${GLUT_LIBRARIES} GL GLU /usr/local/cuda-10.2/samples/common/lib/linux/aarch64/libGLEW.a)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(flashlight PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
