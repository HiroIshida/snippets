#include<iostream>
#include<Eigen/Core>
using namespace Eigen;

using MatrixXdC = Matrix<double, Dynamic, Dynamic, RowMajor>;

int main(){
  {
    std::cout << "=== example 1 ===" << std::endl; 
    double* data = new double [6];
    double* itr = data;
    for(int i=0; i<6; i++){
      *(itr++) = i;
    }
    for(int i=0; i<6; i++){
      std::cout << data[i] << std::endl; 
    }
    auto m = Map<MatrixXd>(data, 2, 3);
    std::cout << m << std::endl; 
  }
  {
    MatrixXd m(2, 3);
    std::cout << m << std::endl; 
  }

  for(int j=0; j<10; j++)
  {
    std::cout << "=== example 2 ===" << std::endl; 
    MatrixXdC m(2, 3);
    std::cout << m << std::endl; 
    double* itr = m.data();
    for(int i=0; i<6; i++){
      *(itr++) = i * 2;
    }
    std::cout << m << std::endl; 
  }
}


