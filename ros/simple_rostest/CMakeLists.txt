cmake_minimum_required(VERSION 3.2)
project(simple_rostest)
add_compile_options(-Wall -Wextra)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  message_runtime
  roscpp
  std_msgs
)

################################################
## Declare ROS messages, services and actions ##
################################################

add_service_files(
  DIRECTORY
  srv
  FILES
  AddTwoInts.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

###################################
## catkin specific configuration ##
###################################

catkin_package(
  CATKIN_DEPENDS
  message_runtime
  std_msgs
)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(
  ${PROJECT_NAME}_add_two_ints_service
  src/add_two_ints_service.cpp
)
target_link_libraries(
  ${PROJECT_NAME}_add_two_ints_service
  ${catkin_LIBRARIES}
)
add_dependencies(
  ${PROJECT_NAME}_add_two_ints_service
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
)

#############
## Install ##
#############

install(
  TARGETS
  ${PROJECT_NAME}_add_two_ints_service
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

#############
## Testing ##
#############

if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)

  add_rostest_gtest(
    add_two_ints_service_client
    test/add_two_ints_service_client.launch
    test/add_two_ints_service_client.cpp
  )
  target_link_libraries(
    add_two_ints_service_client
    ${catkin_LIBRARIES}
  )
  add_dependencies(
    add_two_ints_service_client
    ${PROJECT_NAME}_add_two_ints_service
    ${PROJECT_NAME}_generate_messages_cpp
    ${catkin_EXPORTED_TARGETS}
  )
endif()

