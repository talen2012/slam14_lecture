cmake_minimum_required(VERSION 3.0)

project(useEigen)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# 添加Eigen头文件目录
include_directories("/usr/include/eigen3")

# set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

add_executable(eigenMatrix eigenMatrix.cpp)