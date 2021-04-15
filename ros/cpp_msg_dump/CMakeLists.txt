cmake_minimum_required(VERSION 3.0.2)
project(msgdump)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  nav_msgs
)
catkin_package()

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(${catkin_INCLUDE_DIRS})
add_executable(main main.cpp)
target_link_libraries(main ${catkin_LIBRARIES})
