#include<iostream>
#include<Eigen/Core>

int main(){
  auto a = Eigen::MatrixXd(3, 3);
  a << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  std::cout << a << std::endl;

  a.block(0, 0, 2, 2) << -1, -1, -1, -1;
  std::cout << a << std::endl;

  auto&& b = a.block(0, 0, 2, 2);
  Eigen::Matrix2d c;
  c << -2, -2, -2, -2;
  b = c;
  std::cout << a << std::endl;
}
