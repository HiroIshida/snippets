#include <string>

template <bool T>
struct A{
  typedef typename std::conditional<T, int, std::string>::type ValueType;
  ValueType value;
};

int main(){
  A<false>{"hoge"};
  A<true>{2};
}
