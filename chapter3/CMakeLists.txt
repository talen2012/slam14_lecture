cmake_minimum_required(VERSION 3.0)

project(chapter3)

set(CMAKE_CXX_FLAGS, "${CMAKE_CXX_FLAGS} -std=c++11")

# 加入各子工程
add_subdirectory(useEigen)
add_subdirectory(useGeometry)
add_subdirectory(examples)