cmake_minimum_required(VERSION 3.0.2)

add_compile_options(-std=c++14)

find_package(Eigen3 REQUIRED)
ADD_DEFINITIONS(-DEIGEN_NO_DEBUG)
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(main main.cpp)
