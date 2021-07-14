
cmake_minimum_required(VERSION 3.0.2)
project(perf)

add_compile_options(-std=c++14 -Wall -O3)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  autoware_utils
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(main main.cpp)
target_link_libraries(main
    ${catkin_LIBRARIES}
)
