#include <iostream>
#include <chrono>
extern "C" {
  void bench1(void*);
  void bench2(void*);
  void bench3(void*);
}

void func(){
}

int main(){
  void (*fp)() = func;
  size_t n_inner = 100;
  size_t n_outer = 10000000;
  {
    auto start = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < n_outer; i++){
      bench1((void*)fp);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "bench1: " << duration / (n_outer * n_inner) << " ns" << std::endl;
  }

  {
    auto start = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < n_outer; i++){
      bench2((void*)fp);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "bench2: " << duration / (n_outer * n_inner) << " ns" << std::endl;
  }

  {
    auto start = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < n_outer; i++){
      bench3((void*)fp);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "bench3: " << duration / (n_outer * n_inner) << " ns" << std::endl;
  }


}

