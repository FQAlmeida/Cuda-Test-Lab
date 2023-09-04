cmake_minimum_required(VERSION 3.21)

project(Cuda-Test-Lab)

set(CXX_STANDARD 20)
set(C_STANDARD 17)

set(CMAKE_BUILD_TYPE Debug)

add_subdirectory(Thread-Pool)
add_subdirectory(Convolution)
add_subdirectory(Vector-Add)
