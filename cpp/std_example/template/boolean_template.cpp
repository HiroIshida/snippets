#include <iostream>

template <bool isConst> void hoge();


template <>
void hoge<true>(){
  std::cout << "true" << std::endl;
}

template <>
void hoge<false>(){
  std::cout << "false" << std::endl;
}


int main(){
  hoge<false>();
  hoge<true>();
}
