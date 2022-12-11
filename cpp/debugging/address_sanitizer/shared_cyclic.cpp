#include <memory>

struct A{
  A(std::shared_ptr<A> a) : a_(a) {}
  A() : a_(nullptr) {}
  std::shared_ptr<A> a_;
};

int main() { 
  //auto a = std::make_shared<A>();
  auto aa = std::make_shared<A>(std::make_shared<A>());
  a->a_ = aa;
}
