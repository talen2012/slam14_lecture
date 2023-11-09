cmake_minimum_required(VERSION 3.0)
project(chapter5)

set(CMAKE_BUILD_TYPE Release)
# 添加C++11标准支持
set(CMAKE_CXX_FLAGS "-std=c++14 -O2")

include_directories("../3rdparty/Sophus")

add_subdirectory(imageBasics)
add_subdirectory(stereoVision)
add_subdirectory(rgbd)