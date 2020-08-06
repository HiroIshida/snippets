#include<iostream>
#include<Eigen/Core>
#define print(a) std::cout<<a<<std::endl
using namespace Eigen;

int main(){
  Matrix2d m;
  m << 0, 1, 2, 3;
  Vector2d v;
  v << 1, 2;
  print(m*v);

  MatrixXd M = MatrixXd::Zero(2, 5);
  print(M);
  print(v.transpose() * M);
}
