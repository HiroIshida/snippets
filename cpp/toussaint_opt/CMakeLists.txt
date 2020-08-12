set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

add_subdirectory(Optim)

FILE(GLOB HDRS "*.h" ) # workaround

add_executable(test main.cpp lecture.cpp)
target_link_libraries(test optim core pthread lapack)
