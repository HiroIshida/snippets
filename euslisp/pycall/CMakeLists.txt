cmake_minimum_required(VERSION 3.4 FATAL_ERROR)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#find_package(PythonLibs REQUIRED)

# https://stackoverflow.com/questions/20582270/distribution-independent-libpython-path
set(PYTHON_INCLUDE_DIRS "/usr/include/python2.7")
set(PYTHON_LIBRARIES "/usr/lib/python2.7/config-x86_64-linux-gnu")
link_directories(PYTHON_LIBRARIES)

add_library(main SHARED wrapper.cpp)
include_directories(${PYTHON_INCLUDE_DIRS})
target_link_libraries(main python2.7)
