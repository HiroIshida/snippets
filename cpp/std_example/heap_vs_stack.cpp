#include <iostream>
#include <time.h>
const int N = 100;
using namespace std;

int main(){
  double arr_stack[N];
  double* arr_heap = new double[N];
  int M = 1000000;

  {
    clock_t start = clock();
    for(int i=0; i<M; i++){
      for(int j=0; j<N; j++){
        arr_stack[j];
      }
    }
    std::cout << clock() - start << std::endl; 
  }

      
  {
    clock_t start = clock();
    for(int i=0; i<M; i++){
      for(int j=0; j<N; j++){
        arr_heap[j];
      }
    }
    std::cout << clock() - start << std::endl; 
  }
      
}
