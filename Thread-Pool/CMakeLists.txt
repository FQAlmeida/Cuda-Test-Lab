cmake_minimum_required(VERSION 3.21)

project(Thread-Pool)

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

install(TARGETS Thread-Pool DESTINATION Thread-Pool)
install(FILES ${HEADERS_FILES} DESTINATION Thread-Pool/include)
