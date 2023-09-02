cmake_minimum_required(VERSION 3.21)

project(
    Convolution
)
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

file(
  GLOB SRC_FILES
  CONFIGURE_DEPENDS
  "src/**/*.[hc]pp"
)

file(
  GLOB SRC_FILES_CUDA
  CONFIGURE_DEPENDS
  "src/**/*.cu"
)

add_library(ConvolutionLib src/convolution.cu)

add_executable(Convolution src/main.cpp)
target_link_libraries(Convolution CUDA::cudart)
target_link_libraries(Convolution ConvolutionLib)
# target_sources(Convolution PUBLIC ${SRC_FILES_CUDA})
target_sources(Convolution PUBLIC ${SRC_FILES})
include_directories(${Thread-Pool_SOURCE_DIR})
target_link_libraries(Convolution Thread-Pool)

install(TARGETS Convolution DESTINATION Convolution)
