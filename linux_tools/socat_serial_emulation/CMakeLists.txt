cmake_minimum_required(VERSION 3.10)
project(dynamics_server)

find_package(urdf REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(orocos_kdl REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(server linux_side.cpp)
target_link_libraries(server PRIVATE
    ${urdf_LIBRARIES}
    ${kdl_parser_LIBRARIES}
    ${orocos_kdl_LIBRARIES}
)
target_include_directories(server PUBLIC
    ${urdf_INCLUDE_DIRS}
    ${orocos_kdl_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

add_executable(client stm_side.c)
