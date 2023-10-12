cmake_minimum_required(VERSION 3.0)

project(examples)

include_directories("/usr/include/eigen3")

# set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

add_executable(coordinateTransform coordinateTransform.cpp)

add_executable(plotTrajectory plotTrajectory.cpp)

find_package(Pangolin REQUIRED)
if(Pangolin_FOUND)
    include_directories(${Pangolin_INCLUDE_DIRS}) #实际上Pangolin_INCLUDE_DIRS这个变量内容为空，猜测库文件里已经包含了头文件信息
    target_link_libraries(plotTrajectory ${Pangolin_LIBRARIES})
    message(STATUS "Pangolin_LIBRARIES = ${Pangolin_LIBRARIES}")
else(Pangolin_FOUND)
    message(FATAL_ERROR "Pangolin library not found")
endif(Pangolin_FOUND)