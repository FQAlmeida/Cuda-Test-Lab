cmake_minimum_required(VERSION 3.21)

project(Cuda-Test-Lab)

set(CXX_STANDARD 20)
set(C_STANDARD 17)

add_subdirectory(Thread-Pool)
add_subdirectory(Convolution)
