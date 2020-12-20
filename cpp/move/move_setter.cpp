#include <iostream>
#include <vector>
#include <time.h>
using namespace std;
struct A{
  std::vector<double> data;
  void set_data(vector<double> data){
    this->data = std::move(data);
  }
};

int main(){

  {
    auto a = A();
    auto data = std::vector<double>(10000000, 1);
    clock_t start = clock();
    a.set_data(data);
    clock_t end = clock();
    cout << end - start << endl;
  }

  {
    auto a = A();
    auto data = std::vector<double>(10000000, 1);
    clock_t start = clock();
    a.set_data(std::move(data));
    clock_t end = clock();
    cout << end - start << endl;
  }

}
