cmake_minimum_required(VERSION 3.2)
project(pybind_test VERSION 0.1.0)
add_subdirectory(pybind11)
pybind11_add_module(example example.cpp)
