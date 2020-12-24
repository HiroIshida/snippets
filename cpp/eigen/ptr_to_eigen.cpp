#include<iostream>
#include<Eigen/Core>
using namespace Eigen;
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
}


