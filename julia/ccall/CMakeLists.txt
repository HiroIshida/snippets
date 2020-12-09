cmake_minimum_required(VERSION 2.4.3)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS_INIT} -Wall -fPIC")

add_library(mylib SHARED test.cpp)

