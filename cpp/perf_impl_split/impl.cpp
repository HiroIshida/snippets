#include "header.h"
Tester2::Tester2(int n){ 
  table_.resize(n);
  for(int i=0; i<n; i++){
    table_[i].resize(n);
  }
}

bool Tester2::get_sum(){
  double s = 0;
  for(auto& subtable : table_){
    for(double e : subtable){
      s += e;
    }
  }
  return s;
}
