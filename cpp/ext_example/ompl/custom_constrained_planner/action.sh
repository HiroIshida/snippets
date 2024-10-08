# g++ -v -O3 -Wall -shared -fPIC \
#     $(pkg-config --cflags --libs ompl) \
#     -I/usr/include/eigen3 \
#     $(python3 -m pybind11 --includes) \
#     constrained.cpp \
#     -o constrained$(python3-config --extension-suffix)
# 
# g++ -O3 -Wall -shared -std=c++20 -fPIC \
g++ -g -Wall -shared -std=c++20 -fPIC \
    -I /opt/ros/noetic/include/ompl-1.6 \
    -I /usr/include/eigen3 \
    -I /usr/include/python3.8 \
    -I /home/h-ishida/.local/lib/python3.8/site-packages/pybind11/include \
    constrained.cpp \
    -o constrained.cpython-38-x86_64-linux-gnu.so \
    -Wl,-v \
    -Wl,-rpath,/opt/ros/noetic/lib/x86_64-linux-gnu \
    -L/opt/ros/noetic/lib/x86_64-linux-gnu \
    -Wl,-Bdynamic \
    -lompl \
    -lboost_serialization \
    -lboost_filesystem \
    -lboost_system \
    -lpthread \
    -lode
