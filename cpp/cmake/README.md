## How to pass preprocessor macros
Use `add_definitions` command inside CMake. For instance, if you want to pass `BT_ENABLE_ENET` then:
```CMake
IF(BUILD_CLSOCKET)
 ADD_DEFINITIONS(-DBT_ENABLE_CLSOCKET)
ENDIF()
```
```c++
#ifdef BT_ENABLE_ENET
#include "PhysicsClientUDP_C_API.h"
#endif
```

### commands
`add_library`: generate `libgreetings.a` using `*.cpp` files.
```CMake
add_library(greetings STATIC hello.cpp good_morning.cpp)
```
`target_link_libraries`: use `libhoge.a` (or libhoge.so`) when generating `libgreetings.a`.
```cmake
target_link_libraries(greetings hoge)
```


