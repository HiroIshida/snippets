add_library(hoge SHARED wrapper.cpp)
add_executable(test test.c)
target_link_libraries(test hoge)
