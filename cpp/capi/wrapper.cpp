#include "wrapper.h"
#include<vector>

void* newHoge(int idx, double* arr)
{
  std::vector<double> vec(arr, arr + sizeof(arr)/sizeof(arr[0]));
  Hoge* h_ptr = new Hoge(idx, vec);  
  return (void*)h_ptr;
}
