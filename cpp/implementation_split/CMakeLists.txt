add_library(mylib STATIC a.cpp b.cpp)
add_executable(out main.cpp)
target_link_libraries(out mylib)
