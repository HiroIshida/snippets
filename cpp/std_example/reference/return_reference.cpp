#include <vector>
#include <iostream>

struct Position {
  double x;
  double y;
  Position() : x(0), y(0) {}
  Position(const Position& p) : x(p.x), y(p.y) {
    std::cout << "copy constructor" << std::endl;
  }
};

struct Hoge{
  Hoge(size_t s) : data(s){}
  Position& get(size_t i){ return data[i]; }
  std::vector<Position> data;
};

int main() {
  auto f = Hoge(10);
  double a = f.get(2).x; // copy constructor is not called
  auto b = f.get(2); // copy constructor is called
}
