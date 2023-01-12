#include <iostream>

template <bool Cond>
void hoge(){
  if constexpr(Cond){
    std::cout << "true"  << std::endl;
  }else{
    std::cout << "false" << std::endl;
  }
}

int main(){
  hoge<true>();
  hoge<false>();
}
