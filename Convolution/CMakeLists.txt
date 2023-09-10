cmake_minimum_required(VERSION 3.21)

project(
    Convolution
)

enable_language(CUDA)
enable_language(CXX)
find_package(CUDAToolkit REQUIRED)

file(
  GLOB SRC_FILES
  CONFIGURE_DEPENDS
  "src/**/*.cpp"
)

file(
  GLOB SRC_FILES_CUDA
  CONFIGURE_DEPENDS
  "src/**/*.cu"
)

file(
  GLOB SRC_IMAGE_ALLOC
  CONFIGURE_DEPENDS
  "src/alloc_image_mem/**/*.cpp"
)

add_library(ConvolutionLib src/convolution/convolution.cu src/convolution/convolution.cuh)
target_sources(ConvolutionLib PUBLIC ${SRC_IMAGE_ALLOC})
target_sources(ConvolutionLib PUBLIC ${SRC_FILES_CUDA})
include_directories(${Thread-Pool_SOURCE_DIR})
target_link_libraries(ConvolutionLib Thread-Pool)
target_link_libraries(ConvolutionLib CUDA::cudart)

add_executable(Convolution src/main.cpp)
target_link_libraries(Convolution CUDA::cudart)
target_link_libraries(Convolution ConvolutionLib)
target_sources(Convolution PUBLIC ${SRC_FILES_CUDA})
target_sources(Convolution PUBLIC ${SRC_FILES})
include_directories(${Thread-Pool_SOURCE_DIR})
target_link_libraries(Convolution Thread-Pool)

install(TARGETS Convolution DESTINATION Convolution)
