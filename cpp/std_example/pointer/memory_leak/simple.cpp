#include <iostream>
#include <chrono>
#include <thread>
using namespace std::chrono_literals;

void reserve_memory(){
  double* ptr = new double[10000000];
}

int main(){ // memory leak experiment
  for(int i=0; i<1000000; i++){
    reserve_memory();
  }
  std::cout << "sleeping now" << std::endl;
  std::this_thread::sleep_for(5s);
}
