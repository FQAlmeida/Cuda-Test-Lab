cmake_minimum_required(VERSION 3.21)

project(
    Vector-Add
)

set(CXX_STANDARD 20)
set(CXX_STANDARD_REQUIRED On)
set(C_STANDARD 17)
set(C_STANDARD_REQUIRED On)
set(CUDA_EXTENSIONS On)
set(CUDA_STANDARD 20)
set(CUDA_STANDARD_REQUIRED On)

# set_property(TARGET target PROPERTY CXX_STANDARD 17)
# set_property(TARGET target PROPERTY CXX_STANDARD_REQUIRED On)


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
  "src/alloc_vector_mem/**/*.cpp"
)

add_library(Vector-AddLib src/vector_add/vector_add.cu src/vector_add/vector_add.cuh)
target_sources(Vector-AddLib PUBLIC ${SRC_IMAGE_ALLOC})
target_sources(Vector-AddLib PUBLIC ${SRC_FILES_CUDA})
include_directories(${Thread-Pool_SOURCE_DIR})
target_link_libraries(Vector-AddLib Thread-Pool)
target_link_libraries(Vector-AddLib CUDA::cudart)
set_target_properties(Vector-AddLib PROPERTIES 
  CXX_STANDARD 20  
  CXX_STANDARD_REQUIRED On 
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED On
  CUDA_ARCHITECTURES "52;60;61;70;75;80;86"
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")

add_executable(Vector-Add src/main.cpp)
target_link_libraries(Vector-Add CUDA::cudart)
target_link_libraries(Vector-Add Vector-AddLib)
target_sources(Vector-Add PUBLIC ${SRC_FILES_CUDA})
target_sources(Vector-Add PUBLIC ${SRC_FILES})
include_directories(${Thread-Pool_SOURCE_DIR})
target_link_libraries(Vector-Add Thread-Pool)

install(TARGETS Convolution DESTINATION Convolution)
