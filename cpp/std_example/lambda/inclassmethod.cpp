#include<iostream>
#define print(a) std::cout<<a<<std::endl
using namespace std;
class Hoge
{
  public:
    Hoge(int h) : i(h) {}
    int i;
    void method_using_lambda();
    void incf(){i++;}
};

void Hoge::method_using_lambda(){
  auto lambda = [this](int j){
    this->incf();
    return (this->i + j);
  };
  print(lambda(1));
}

int main(){
  Hoge h(0);
  h.method_using_lambda();
  h.incf();
  h.method_using_lambda();
}


