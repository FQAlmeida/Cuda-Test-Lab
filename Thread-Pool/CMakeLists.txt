cmake_minimum_required(VERSION 3.21)

project(Thread-Pool)

set(CXX_STANDARD 20)
set(C_STANDARD 17)

file(GLOB SRC_FILES
    CONFIGURE_DEPENDS
    "src/*.cpp"
)
file(GLOB HEADERS_FILES
    CONFIGURE_DEPENDS
    "src/*.hpp"
)

add_library(Thread-Pool SHARED ${SRC_FILES} ${HEADERS_FILES})
target_include_directories(Thread-Pool PUBLIC include)
set_target_properties(Thread-Pool PROPERTIES 
  CXX_STANDARD 20  
  CXX_STANDARD_REQUIRED On 
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")

install(TARGETS Thread-Pool DESTINATION Thread-Pool)
install(FILES ${HEADERS_FILES} DESTINATION Thread-Pool/include)
