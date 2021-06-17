When I split the header and implementation in c++, I found that significant slow-down occurs in some situation compared to the header-only one. The minumum working example is like below. In the case below, `get_sum` function is implemented in the header file and `get_sum2`, which is exactly the same implementation as `get_sum`, is implemented in `impl.cpp`. The result shows that time duration to execute is 0.001 [msec] for `get_sum` and 1066.84 [msec] for `get_sum2` (see `main.cpp`), though both have the same implementation. Could anyone explain why this happens?

The following codes can be found [here](https://github.com/HiroIshida/snippets/tree/master/cpp/perf_impl_split).

The header file `header.h`
```c++
#include<vector>
#include<iostream>

using namespace std;
class Tester
{
  public:
    Tester(int n){ 
      table_.resize(n);
      for(int i=0; i<n; i++){
        table_[i].resize(n);
      }
    }

    bool get_sum(){
      double s = 0;
      for(auto& subtable : table_){
        for(double e : subtable){
          s += e;
        }
      }
      return s;
    }
    bool get_sum2(); // definition is in impl.cpp
  private:
    vector<vector<double>> table_;
};

```

The implementation file `impl.cpp`
```c++
bool Tester::get_sum2(){
  double s = 0;
  for(auto& subtable : table_){
    for(double e : subtable){
      s += e;
    }
  }
  return s;
}
```

The main file for benchmarking `main.cpp`
```cpp
#include "header.h"
int main(){
  int size = 30000;
  auto tester = Tester(size);
  std::cout << "Finish constructing the tester" << std::endl; 
  {
    clock_t start = clock();
    tester.get_sum();
    clock_t end = clock();
    std::cout << (end - start)/1000.0 << " [msec]" << std::endl; 
  }

  {
    clock_t start = clock();
    tester.get_sum2();
    clock_t end = clock();
    std::cout << (end - start)/1000.0 << " [msec]" << std::endl; 
  }
}
```

cmake script used `CmakeLists.txt` (note that build with release mode)
```
cmake_minimum_required(VERSION 3.4 FATAL_ERROR)
set(CMAKE_BUILD_TYPE Release)
add_executable(main main.cpp impl.cpp)
```
