cmake_minimum_required(VERSION 3.0)
project(imageBasics)

# 寻找OpenCV库
find_package(OpenCV REQUIRED)
# 添加头文件
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(imageBasics imageBasics.cpp)
# 链接OpenCV库
target_link_libraries(imageBasics ${OpenCV_LIBS})

add_executable(undistortImage undistortImage.cpp)
target_link_libraries(undistortImage ${OpenCV_LIBS})