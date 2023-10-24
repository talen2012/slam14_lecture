cmake_minimum_required(VERSION 3.0)
project(stereoVision)

# 包含eigen3头文件目录
include_directories("/usr/include/eigen3")
add_executable(stereoVision stereoVision.cpp)

# 添加Pangolin库并链接
find_package(Pangolin REQUIRED)
# 添加OpenCV库并链接
find_Package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(stereoVision ${Pangolin_LIBRARIES} ${OpenCV_LIBS})


