currently ament has really scarse documentation !!!! 
see: https://autowarefoundation.gitlab.io/autoware.auto/AutowareAuto/contributor-guidelines.html
```cmake
cmake_minimum_required(VERSION 3.5)
project(composition_example)
 
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()
 
ament_auto_add_library(listener_node SHARED src/listener_node.cpp)
autoware_set_compile_options(listener_node)
rclcpp_components_register_nodes(listener_node "composition_example::ListenerNode")
 
ament_auto_add_executable(listener_node_exe src/listener_main.cpp)
autoware_set_compile_options(listener_node_exe)
 
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()
 
ament_auto_package()
```
