#include<vector>
#include<iostream>

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
    bool get_sum2(); // definition is in impl.cpp
  private:
    vector<vector<double>> table_;
};

