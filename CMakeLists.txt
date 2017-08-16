cmake_minimum_required(VERSION 3.6)
project(AEMLKit)

set(CMAKE_CXX_STANDARD 11)

find_package(Shark REQUIRED)
include(${SHARK_USE_FILE})

set(DEMO_SOURCE_FILES demo.cpp AEMLKit.cpp)
add_executable(demo ${DEMO_SOURCE_FILES})
target_link_libraries(demo ${SHARK_LIBRARIES})