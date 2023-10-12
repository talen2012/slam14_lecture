cmake_minimum_required(VERSION 3.0)
project(useSophus)

include_directories("/usr/include/eigen3")
find_package(Sophus REQUIRED)
include_directories("../3rdparty/Sophus")

add_executable(useSophus useSophus.cpp)