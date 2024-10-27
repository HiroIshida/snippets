#include <chrono>
#include <iostream>
extern "C" void movsd_xmm_xmm_bench();
extern "C" void movsd_xmm_stack_bench();

int main() {
  {
    std::cout << "movsd xmm <=> xmm\n";
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 1000; i++){
      movsd_xmm_xmm_bench();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s\n";
  }
  {
    std::cout << "movsd xmm <=> stack\n";
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 1000; i++){
      movsd_xmm_stack_bench();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s\n";
  }
}
