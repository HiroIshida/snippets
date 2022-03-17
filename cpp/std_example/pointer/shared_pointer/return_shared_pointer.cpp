#include <memory>
#include <iostream>

using namespace std;

struct Ishida
{
  int value;
};

std::shared_ptr<Ishida> test(){
  const auto a = std::make_shared<Ishida>(Ishida{10});
  cout << a.use_count() << endl;
  return a;
}

int main(){
  const auto ishi = test();
  std::cout << ishi->value << std::endl;
  cout << ishi.use_count() << endl;
  const auto hoge = ishi;
  cout << ishi.use_count() << endl;

}
