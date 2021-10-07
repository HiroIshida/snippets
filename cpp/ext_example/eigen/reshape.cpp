#include<iostream>
#include<Eigen/Core>
#define print(a) std::cout<<a<<std::endl
using namespace Eigen;

int main(){
  MatrixXd m = MatrixXd::Zero(2, 4);
  double* itr = m.data();
  for(int i=0; i<8; i++){
    itr[i] = i;
  }
  std::cout << m << std::endl; 
  // m.reshaped(8, 2); // only after version 3.4
  MatrixXd mm = Map<MatrixXd>(m.data(),4 ,2);
  std::cout << mm << std::endl; 
}
