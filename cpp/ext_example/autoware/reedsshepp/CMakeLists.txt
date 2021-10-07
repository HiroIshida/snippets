cmake_minimum_required(VERSION 3.0.2)
project(reedssheepp)

add_compile_options(-std=c++14 -Wall -O3)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  std_msgs
  tf2
  tf2_geometry_msgs
  autoware_utils
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES freespace_planning_algorithms
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(main.cpp)
add_executable(main main.cpp)
target_link_libraries(main
    freespace_planning_algorithms
    ${catkin_LIBRARIES}
)
