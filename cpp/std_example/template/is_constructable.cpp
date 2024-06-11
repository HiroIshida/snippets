#include <iostream>
#include <type_traits>

template <typename T, typename = std::enable_if_t<std::is_constructible<T, int, double>::value>>
void func() {
    T obj(42, 3.14);
}

struct A {
  int x;
  double y;
  A(int x, double y) : x(x), y(y) {}
};

class B {
public:
  B(int x) {}
};

int main() {
    func<A>(); // This should work because MyClass(int, double) constructor exists
    // func<B>();  // Cannot compile
}
