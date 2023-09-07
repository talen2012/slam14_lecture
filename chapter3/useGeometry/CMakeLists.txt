cmake_minimum_required(VERSION 3.0)

project(useGeometry)

include_directories("/usr/include/eigen3")

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

add_executable(eigenGeometry eigenGeometry.cpp)

