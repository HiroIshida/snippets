#include "header.h"

bool Tester::get_sum2(){
  double s = 0;
  for(auto& subtable : table_){
    for(double e : subtable){
      s += e;
    }
  }
  return s;
}
