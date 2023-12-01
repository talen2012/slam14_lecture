cmake_minimum_required(VERSION 3.0)
project(chapter7)

set(CMAKE_BUILD_TYPE Release)
add_definitions("-DENABLE_SSE")
# set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 -msse4")
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# 查找OpenCV3的包
find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(orb_cv orb_cv.cpp)
target_link_libraries(orb_cv ${OpenCV_LIBS})

add_executable(orb_self orb_self.cpp)
target_link_libraries(orb_self ${OpenCV_LIBS})

add_executable(pose_estimation_2d2d pose_estimation_2d2d.cpp)
target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS})

add_executable(triangulation triangulation.cpp)
target_link_libraries(triangulation ${OpenCV_LIBS})

# 查找Eigen3的包
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})
message(${EIGEN3_INCLUDE_DIRS})

# 由于本机并未安装Sophus,直接将本项目下该包的相对路径添加到头文件目录
# Sophus和Eigen一样只有头文件
include_directories("../3rdparty/Sophus")

# 查找G2O包
# 在/usr/local/lib/cmake/g2o目录下有一个g2oConfig.cmake文件
# 但文件名并不是G2OConfig.cmake或者g2o_config.cmake且文件内容只有几行
# 必须手动添加FindG2O.cmake文件，并且指定CMAKE_MODULE_PATH
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIR}) # 注意结尾是DIR不是DIRS

add_executable(pose_estimation_3d2d pose_estimation_3d2d.cpp)
target_link_libraries(pose_estimation_3d2d ${OpenCV_LIBS} g2o_core g2o_stuff)
