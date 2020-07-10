## How to pass preprocessor macros
Use `add_definitions` command inside CMake. For instance, if you want to pass `BT_ENABLE_ENET` then:
```CMake
IF(BUILD_CLSOCKET)
 ADD_DEFINITIONS(-DBT_ENABLE_CLSOCKET)
ENDIF()
```c++
#ifdef BT_ENABLE_ENET
#include "PhysicsClientUDP_C_API.h"
#endif
```
```
