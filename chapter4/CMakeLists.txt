cmake_minimum_required(VERSION 3.0)
project(useSophus)

# set(CMAKE_BUILD_TYPE DEBUG)
include_directories("/usr/include/eigen3")

include_directories("../3rdparty/Sophus")

add_executable(useSophus useSophus.cpp)

add_subdirectory(example)