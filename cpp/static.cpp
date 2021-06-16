#include <iostream>

int main(){
  int hoge = -1;
  for(int i=0; i<10; i++){ 
    for(int j=0; j<10; j++){
      static int counter = 0;
      counter++;
      hoge = counter;
    }
  }
  std::cout << hoge << std::endl;  // 100
}
