set(CMAKE_CXX_STANDARD 14)

add_subdirectory(Core)

add_library(optim 
  optimization.cpp 
  newton.cpp 
  gradient.cpp 
  convert.cpp 
  lagrangian.cpp
  benchmarks.cpp)

add_executable(test main.cpp)
target_link_libraries(test optim core pthread)
include_directories(./)
