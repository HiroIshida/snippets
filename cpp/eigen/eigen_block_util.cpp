#include<iostream>
#include<Eigen/Core>
#include <cassert>
#include <time.h>

using namespace Eigen;
using namespace std;

struct SubMatrix
{
  double* _data; // beginning of the block matrix
  int _i_begin;
  int _j_begin;

  int _n_block;
  int _m_block;

  int _n_whole;
  int _m_whole;

  SubMatrix(Eigen::MatrixXd& mat, int i_begin, int j_begin, int n, int m) : 
    _data(mat.data()),
    _i_begin(i_begin), _j_begin(j_begin),
    _n_block(n), _m_block(m),
    _n_whole(mat.rows()), _m_whole(mat.cols()) {}

  inline int get_idx(int i, int j){
    assert(i<_n_whole && "out of index");
    assert(j<_m_whole && "out of index");

    int idx = _n_whole * (j+_j_begin) + (i+_i_begin);
    return idx;
  }

  double& get(int i, int j){
    return _data[this->get_idx(i, j)];
  }
};

int main(){
  {
    MatrixXd m(8, 6);
    auto bm = SubMatrix(m, 2, 2, 2, 2);
    bm.get(0, 0) = 1;
    bm.get(0, 1) = 2;
    bm.get(1, 0) = 3;
    bm.get(1, 1) = 4;
    std::cout << m << std::endl; 
  }

  // check how does it slow
  int n_itr = 1000;
  int N = 100;
  int M = 100;
  {
    clock_t start = clock();
    for(int k=0; k<n_itr; k++){

      MatrixXd m(N, M);
      auto bm = SubMatrix(m, 0, 0, N, M);

      for(int i=0; i<N; i++){
        for(int j=0; j<M; j++){
          bm.get(i, j) = 1.0;
        }
      }
    }
    clock_t end = clock();
    cout << end - start << endl;
  }

  {
    clock_t start = clock();
    for(int k=0; k<n_itr; k++){

      MatrixXd m(N, M);
      double* ptr = m.data();
      for(int i=0; i<N; i++){
        for(int j=0; j<M; j++){
          ptr[j*N+i] = 1.0;
        }
      }
    }
    clock_t end = clock();
    cout << end - start << endl;
  }
  // bm.get(1, 10) = 4; assertion error
}
