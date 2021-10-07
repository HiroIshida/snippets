#include<iostream>
#include<Eigen/Core>
#include <time.h>
using namespace Eigen;
using namespace std;

int main(){
  // check how does it slow
  int n_itr = 1000;
  int N = 100;
  {
    clock_t start = clock();
    for(int k=0; k<n_itr; k++){

      MatrixXd m(N, N);
      for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
          m(i, j) = 1.0;
        }
      }
    }
    clock_t end = clock();
    cout << end - start << endl;
  }
  {
    clock_t start = clock();
    for(int k=0; k<n_itr; k++){
      MatrixXd m(N, N);
      double* ptr = m.data();
      for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
          ptr[j*N+i] = 1.0;
        }
      }
    }
    clock_t end = clock();
    cout << end - start << endl;
  }
}
