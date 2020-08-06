cmake_minimum_required(VERSION 3.2)
project(pybind_test VERSION 0.1.0)

find_package(Eigen3 REQUIRED)
ADD_DEFINITIONS(-DEIGEN_NO_DEBUG)

include_directories(${EIGEN3_INCLUDE_DIR})
add_subdirectory(pybind11)
pybind11_add_module(example example.cpp)
