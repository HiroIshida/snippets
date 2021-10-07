#include <iostream>
#include <variant>

int main(){
  std::variant<int, double> a;

  // Creates a new value in-place, in an existing variant object
  a.template emplace<int>(1); // what's this???
  a.emplace<int>(1);

  std::cout << std::get<int>(a) << std::endl; 
}
