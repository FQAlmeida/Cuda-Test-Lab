cmake_minimum_required(VERSION 3.21)

project(Cuda-Test-Lab)

find_package(benchmark REQUIRED)

set(CXX_STANDARD 20)
set(C_STANDARD 17)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
set(CMAKE_CUDA_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --threads 0")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -m64")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin g++")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_50,code=sm_50")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_52,code=sm_52")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_60,code=sm_60")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_61,code=sm_61")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_70,code=sm_70")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=sm_75")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_80,code=sm_80")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++20")

add_subdirectory(Thread-Pool)
add_subdirectory(Convolution)
add_subdirectory(Vector-Add)
