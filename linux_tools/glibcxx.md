### Correspondence between glibc version and libstdc++ version
see https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html


### backward compatibility
basically backward compatible
https://stackoverflow.comquestions/11107263/how-compatible-are-different-versions-of-glibc#:~:text=In%20short%2C%20glibc%20is%20backward,compatible%2C%20not%20forward%2Dcompatible.
GCC 4.3.0: libstdc++.so.6.0.10
GCC 4.4.0: libstdc++.so.6.0.11
GCC 4.4.1: libstdc++.so.6.0.12
GCC 4.4.2: libstdc++.so.6.0.13
GCC 4.5.0: libstdc++.so.6.0.14
GCC 4.6.0: libstdc++.so.6.0.15
GCC 4.6.1: libstdc++.so.6.0.16
GCC 4.7.0: libstdc++.so.6.0.17
GCC 4.8.0: libstdc++.so.6.0.18
GCC 4.8.3: libstdc++.so.6.0.19
GCC 4.9.0: libstdc++.so.6.0.20
GCC 5.1.0: libstdc++.so.6.0.21
GCC 6.1.0: libstdc++.so.6.0.22
GCC 7.1.0: libstdc++.so.6.0.23
GCC 7.2.0: libstdc++.so.6.0.24
GCC 8.1.0: libstdc++.so.6.0.25
GCC 9.1.0: libstdc++.so.6.0.26
GCC 9.2.0: libstdc++.so.6.0.27
GCC 9.3.0: libstdc++.so.6.0.28
GCC 10.1.0: libstdc++.so.6.0.28
GCC 11.1.0: libstdc++.so.6.0.29

### Some command to inspect

```
ldconfig -p |grep stdc++
bstdc++.so.6 (libc6,x86-64) => /lib/x86_64-linux-gnu/libstdc++.so.6
```

```
strings /lib/x86_64-linux-gnu/libstdc++.so.6|grep LIBCXX

GLIBCXX_3.4
GLIBCXX_3.4.1
GLIBCXX_3.4.2
GLIBCXX_3.4.3
GLIBCXX_3.4.4
GLIBCXX_3.4.5
GLIBCXX_3.4.6
GLIBCXX_3.4.7
GLIBCXX_3.4.8
GLIBCXX_3.4.9
GLIBCXX_3.4.10
GLIBCXX_3.4.11
GLIBCXX_3.4.12
GLIBCXX_3.4.13
GLIBCXX_3.4.14
GLIBCXX_3.4.15
GLIBCXX_3.4.16
GLIBCXX_3.4.17
GLIBCXX_3.4.18
GLIBCXX_3.4.19
GLIBCXX_3.4.20
GLIBCXX_3.4.21
GLIBCXX_3.4.22
GLIBCXX_3.4.23
GLIBCXX_3.4.24
GLIBCXX_3.4.25
GLIBCXX_3.4.26
GLIBCXX_3.4.27
GLIBCXX_3.4.28
GLIBCXX_DEBUG_MESSAGE_LENGTH
```
```c++
// libdatestamp.cxx
#include <cstdio>

int main(int argc, char* argv[]){
#ifdef __GLIBCPP__
    std::printf("GLIBCPP: %d\n",__GLIBCPP__);
#endif
#ifdef __GLIBCXX__
    std::printf("GLIBCXX: %d\n",__GLIBCXX__);
#endif
   return 0;
}
```
```
$ g++ libdatestamp.cxx -o libdatestamp
$ ./libdatestamp
GLIBCXX: 20101208
```

