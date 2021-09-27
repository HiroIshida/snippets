cmake_minimum_required(VERSION 3.8)
project(sample)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
add_compile_options(-g)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_srvs REQUIRED)

add_executable(serverclient serverclient.cpp)
ament_target_dependencies(serverclient "rclcpp" "std_srvs")
install(TARGETS serverclient DESTINATION lib/${PROJECT_NAME})

add_executable(server server.cpp)
ament_target_dependencies(server "rclcpp" "std_srvs")
install(TARGETS server DESTINATION lib/${PROJECT_NAME})

add_executable(client client.cpp)
ament_target_dependencies(client "rclcpp" "std_srvs")
install(TARGETS client DESTINATION lib/${PROJECT_NAME})

ament_package()
