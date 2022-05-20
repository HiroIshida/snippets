#include <cmath>
#include <iostream>
#include <chrono>

using namespace std::chrono;

inline double myhypot(double a, double b) {
  return a * a + b * b;
}

int main() {
  size_t N = 1000000000;

  {
    auto start = high_resolution_clock::now();
    double a = 0.0;
    for (size_t i = 0; i < N; ++i) {
      a += std::hypot(1.0, 2.0);
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    std::cout << a << std::endl; // 704 [msec]
    std::cout << duration.count() << std::endl;
  }

  {
    auto start = high_resolution_clock::now();
    double a = 0.0;
    for (size_t i = 0; i < N; ++i) {
      a += myhypot(1.0, 2.0);
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    std::cout << a << std::endl; // 694 [msec]
    std::cout << duration.count() << std::endl;
  }

}
