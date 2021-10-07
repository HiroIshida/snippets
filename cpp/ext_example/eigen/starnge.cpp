#include<iostream>
#include<Eigen/Core>
using namespace Eigen;

int main(){
  {
    std::cout << "scope1" << std::endl; 
    MatrixXd m(2, 2);
    std::cout << m << std::endl; 
    m << 1, 2, 3, 4;
    std::cout << m << std::endl; 
  }
  auto a = new double[100];
  {
    std::cout << "scope2" << std::endl; 
    MatrixXd k(2, 2);
    std::cout << k << std::endl; 
    k << 1, 2, 3, 4;
    std::cout << k << std::endl; 
  }
}
