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
  GLOB HEADER_FILES
  CONFIGURE_DEPENDS
  "src/**/*.hpp"
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
target_sources(ConvolutionLib PUBLIC ${SRC_FILES_CUDA} ${HEADER_FILES})
include_directories(${Thread-Pool_SOURCE_DIR})
target_link_libraries(ConvolutionLib Thread-Pool)
target_link_libraries(ConvolutionLib CUDA::cudart)
target_link_libraries(ConvolutionLib benchmark::benchmark)

add_executable(Convolution src/benchmark.cpp)
target_link_libraries(Convolution CUDA::cudart)
target_link_libraries(Convolution ConvolutionLib)
target_sources(Convolution PUBLIC ${SRC_FILES_CUDA})
target_sources(Convolution PUBLIC ${SRC_FILES} ${HEADER_FILES})
include_directories(${Thread-Pool_SOURCE_DIR})
target_link_libraries(Convolution Thread-Pool)
target_link_libraries(Convolution benchmark::benchmark)

add_executable(ConvolutionTest src/test.cpp)
target_link_libraries(ConvolutionTest CUDA::cudart)
target_link_libraries(ConvolutionTest ConvolutionLib)
target_sources(ConvolutionTest PUBLIC ${SRC_FILES_CUDA})
target_sources(ConvolutionTest PUBLIC ${SRC_FILES} ${HEADER_FILES})
include_directories(${Thread-Pool_SOURCE_DIR})
target_link_libraries(ConvolutionTest Thread-Pool)

add_executable(ConvolutionMain src/main.cpp)
target_link_libraries(ConvolutionMain CUDA::cudart)
target_link_libraries(ConvolutionMain ConvolutionLib)
target_sources(ConvolutionMain PUBLIC ${SRC_FILES_CUDA})
target_sources(ConvolutionMain PUBLIC ${SRC_FILES} ${HEADER_FILES})
include_directories(${Thread-Pool_SOURCE_DIR})
target_link_libraries(ConvolutionMain Thread-Pool)

install(TARGETS Convolution DESTINATION Convolution)
