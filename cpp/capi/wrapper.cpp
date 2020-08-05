#include "wrapper.h"
#include <vector>
#include <string>
#include <iostream>

class Hoge
{
  public:
    Hoge(int idx_, std::vector<double> arr_) : idx(idx_), arr(arr_){
      std::cout << "class constructor is called" << std::endl;
    };

    int heck(int a){
      std::cout << "hogehoge" << std::endl;
      return (idx + a);}
    int idx;
    std::vector<double> arr;
};

void* newHoge(int idx, double* arr)
{
  std::vector<double> vec(arr, arr + sizeof(arr)/sizeof(arr[0]));
  Hoge* h_ptr = new Hoge(idx, vec);  
  return (void*)h_ptr;
}

int operation(void* obj, int a){
  Hoge* obj_= static_cast<Hoge*>(obj);
  return obj_->heck(a);
}
