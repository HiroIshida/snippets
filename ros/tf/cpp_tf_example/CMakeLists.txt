cmake_minimum_required(VERSION 3.0.2)
project(tmp)

find_package(catkin REQUIRED 
    COMPONENTS 
    tf
    )

include_directories(
 ${catkin_INCLUDE_DIRS}
)
add_executable(tftest main.cpp)
target_link_libraries(tftest
  ${catkin_LIBRARIES}
)
