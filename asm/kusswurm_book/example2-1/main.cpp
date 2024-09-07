#include <iostream>

extern "C" int AddSubI32(int a, int b, int c, int d);

int main(){
  int a = 10;
  int b = 2;
  int c = 3;
  int d = 4;
  int result = AddSubI32(a, b, c, d);
  std::cout << "Result: " << result << std::endl;
}
