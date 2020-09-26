cmake_minimum_required(VERSION 3.2)
set(CMAKE_BUILD_TYPE Release)
project(pybind_test VERSION 0.1.0)

include_directories(${EIGEN3_INCLUDE_DIR})
add_subdirectory(pybind11)
set(LIBRARY_NAME example)
pybind11_add_module(${LIBRARY_NAME} example.cpp)

# running python script inside cmake
# https://stackoverflow.com/questions/49053544/how-do-i-run-a-python-script-every-time-in-a-cmake-build
# https://stackoverflow.com/questions/122327/how-do-i-find-the-location-of-my-python-site-packages-directory
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import site; print(site.getsitepackages()[0], end='')"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES_INSTALL_DIR
    )
message(${PYTHON_SITE_PACKAGES_INSTALL_DIR})
install(TARGETS ${LIBRARY_NAME} DESTINATION ${PYTHON_SITE_PACKAGES_INSTALL_DIR})
