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
