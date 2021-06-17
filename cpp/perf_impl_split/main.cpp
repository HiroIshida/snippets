#include<vector>
#include<iostream>
#include "header.h"

using namespace std;
class Tester
{
  public:
    Tester(int n){ 
      table_.resize(n);
      for(int i=0; i<n; i++){
        table_[i].resize(n);
      }
    }
    bool get_sum(){
      double s = 0;
      for(auto& subtable : table_){
        for(double e : subtable){
          s += e;
        }
      }
      return s;
    }

  private:
    vector<vector<double>> table_;
};

int main(){
  int size = 30000;
  {
    auto tester = Tester(size);
    clock_t start = clock();
    tester.get_sum();
    clock_t end = clock();
    std::cout << (end - start)/1000.0 << " [msec]" << std::endl; 
  }

  {
    auto tester2 = Tester2(size);
    clock_t start = clock();
    tester2.get_sum();
    clock_t end = clock();
    std::cout << (end - start)/1000.0 << " [msec]" << std::endl; 
  }
}
