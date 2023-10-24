cmake_minimum_required(VERSION 3.0)
project(joinMap)

# sophus的头文件包含，放在了chapter5的CMakeLists.txt

find_package(Pangolin REQUIRED)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(joinMap joinMap.cpp)
target_link_libraries(joinMap ${OpenCV_LIBS} ${Pangolin_LIBRARIES})