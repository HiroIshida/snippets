#include <chrono>
#include <iostream>
#include <cmath>
extern "C" void movsd_xmm_xmm_bench();
extern "C" void movsd_xmm_stack_bench();
extern "C" void movapd_xmm_xmm_bench();
extern "C" void movapd_xmm_stack_bench();
extern "C" void movq_xmm_rax_bench();
extern "C" void vmovq_xmm_rax_bench();
extern "C" void addsd_operation_bench();
extern "C" void mulsd_operation_bench();

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

  {
    std::cout << "movq xmm <=> rax\n";
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < n_here; i++){
      movq_xmm_rax_bench();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time per instr: " << elapsed.count() / n * 1e9 << " ns\n";
  }

  {
    std::cout << "vmovq xmm <=> rax\n";
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < n_here; i++){
      vmovq_xmm_rax_bench();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time per instr: " << elapsed.count() / n * 1e9 << " ns\n";
  }


  {
    std::cout << "addsd operation\n";
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < n_here; i++){
      addsd_operation_bench();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time per instr: " << elapsed.count() / n * 1e9 << " ns\n";
  }
  {
    std::cout << "mulsd operation\n";
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < n_here; i++){
      mulsd_operation_bench();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time per instr: " << elapsed.count() / n * 1e9 << " ns\n";
  }
  {
    // sqrt
    std::cout << "sqrt operation\n";
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < n; i++){
      volatile double a = 1.0;
      a = std::sqrt(a);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time per instr: " << elapsed.count() / n * 1e9 << " ns\n";
  }
  {
    // sin
    std::cout << "sin operation\n";
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < n; i++){
      volatile double a = 1.0;
      a = std::sin(a);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time per instr: " << elapsed.count() / n * 1e9 << " ns\n";
  }
}
