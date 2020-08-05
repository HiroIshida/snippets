#include <vector>
#include <string>
class Hoge
{
  public:
    Hoge(int idx_, std::vector<double> arr_) : idx(idx_), arr(arr_){};
    int idx;
    std::vector<double> arr;
};
