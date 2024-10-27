#include <chrono>
#include <iostream>
extern "C" void movsd_xmm_xmm_bench();
extern "C" void movsd_xmm_stack_bench();
extern "C" void movapd_xmm_xmm_bench();
extern "C" void movapd_xmm_stack_bench();

int main() {
  int n_here = 1000;
  int n = 200000 * n_here;  // 200000 inside the loop (2 for round-trip)
  {
    std::cout << "movsd xmm <=> xmm\n";
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < n_here; i++){
      movsd_xmm_xmm_bench();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time per instr: " << elapsed.count() / n * 1e9 << " ns\n";
  }
  {
    std::cout << "movsd xmm <=> stack\n";
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < n_here; i++){
      movsd_xmm_stack_bench();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time per instr: " << elapsed.count() / n * 1e9 << " ns\n";
  }

  {
    std::cout << "movapd xmm <=> xmm\n";
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < n_here; i++){
      movapd_xmm_xmm_bench();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time per instr: " << elapsed.count() / n * 1e9 << " ns\n";
  }
  {
    std::cout << "movapd xmm <=> stack\n";
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < n_here; i++){
      movapd_xmm_stack_bench();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time per instr: " << elapsed.count() / n * 1e9 << " ns\n";
  }
}
