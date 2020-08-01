cmake_minimum_required(VERSION 2.4.3)
set(CMAKE_CXX_STANDARD 11)
PROJECT(MESHCAT_EXAMPLE)

FILE(GLOB TDS_HDRS "*.h" )

message("USING third_party/zmq")
SET(cppzmq_FOUND TRUE)
SUBDIRS(third_party/zeromq)
set(cppzmq_INCLUDE_DIRS third_party/zeromq/include )
include_directories(${cppzmq_INCLUDE_DIRS})
add_definitions(-DZMQ_STATIC)


find_package(nlohmann_json)
IF (nlohmann_json_FOUND)
ELSE()
    SET(nlohmann_json_FOUND TRUE)
    set(nlohmann_json_INCLUDE_DIRS third_party/nlohmann/include)
    include_directories(${nlohmann_json_INCLUDE_DIRS})
ENDIF()

set(MESHCAT_LIBRARIES zmq) # droff cpp_base64 for now
add_executable(test example.cpp ${TDS_HDRS})
target_link_libraries(test  ${MESHCAT_LIBRARIES})
