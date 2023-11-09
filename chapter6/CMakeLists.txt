cmake_minimum_required(VERSION 3.5)
project(chapter6)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

# Eigen
include_directories("/usr/include/eigen3")
# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(gaussNewton gaussNewton.cpp)
target_link_libraries(gaussNewton ${OpenCV_LIBS})

# Ceres 这里安装的是1.14.0版本，安装最新的2.2版本无法编译通过
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

add_executable(ceresCurveFitting ceresCurveFitting.cpp)
target_link_libraries(ceresCurveFitting ${OpenCV_LIBS} ${CERES_LIBRARIES})

# 生成修改模型后的Ceres优化程序
add_executable(ceresCurveFitting2 ceresCurveFitting2.cpp)
target_link_libraries(ceresCurveFitting2 ${OpenCV_LIBS} ${CERES_LIBRARIES})

# g2o库的FindG2O.cmake文件并没有安装，要自己从源码中添加到工程目录
# 并告诉CMAKE其路径
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
# g2o 2020.12.23版
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

add_executable(g2oCurveFitting g2oCurveFitting.cpp)
target_link_libraries(g2oCurveFitting ${OpenCV_LIBS} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY})

# 生成修改模型后的G2O优化程序
add_executable(g2oCurveFitting2 g2oCurveFitting2.cpp)
target_link_libraries(g2oCurveFitting2 ${OpenCV_LIBS} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY})