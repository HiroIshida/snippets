#include <vector>
#include <string>
#include <iostream>
using namespace std;
class Hoge
{
  public:
    Hoge(int idx_, std::vector<double> arr_) : idx(idx_), arr(arr_){
      cout << "class constructor is called" << endl;
      cout << idx << endl;
    };
    int idx;
    std::vector<double> arr;
};
