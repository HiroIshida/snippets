#include<iostream>
#include<Eigen/Core>
#include <cassert>
#include <time.h>

using namespace Eigen;
using namespace std;

struct SubMatrix // coll major (same as eigen)
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

  SubMatrix(Eigen::MatrixXd& mat) :
    _data(mat.data()),
    _i_begin(0), _j_begin(0),
    _n_block(mat.rows()), _m_block(mat.cols()),
    _n_whole(_n_block), _m_whole(_m_block) {}

  SubMatrix(double* data, int i_begin, int j_begin, int n, int m, int n_whole, int m_whole) :
    _data(data), _i_begin(i_begin), _j_begin(j_begin), _n_block(n), _m_block(m), 
    _n_whole(n_whole), _m_whole(m_whole) {}

  inline int get_idx(int i, int j){
    assert(i<_n_whole && "out of index");
    assert(j<_m_whole && "out of index");

    int idx = _n_whole * (j+_j_begin) + (i+_i_begin);
    return idx;
  }

  double& get(int i, int j){
    return _data[this->get_idx(i, j)];
  }

  SubMatrix block(int i, int j, int n, int m){
    int i_begin_new = _i_begin + i;
    int j_begin_new = _j_begin + j;
    SubMatrix mat = {_data, i_begin_new, j_begin_new, _n_block, _m_block, _n_whole, _m_whole};
    return mat;
  }

  SubMatrix slice(int i){ // we consider matrix is coll major. 
    return this->block(0, i, _n_block, 1);
  }

  double& operator() (int i, int j){
    return _data[this->get_idx(i, j)];
  }

  double& operator[] (int i){ // access to sliced matrix
    return _data[this->get_idx(i, 0)];
  }
};

int main(){
  {
    MatrixXd m_(8, 6);
    SubMatrix m(m_); 
    auto sm = m.block(1, 1, 2, 2);
    sm(0, 0) = 1;
    sm(0, 1) = 2;
    sm(1, 0) = 3;
    sm(1, 1) = 4;

    SubMatrix vec = m.slice(4);
    for(int i=0; i<8; i++){
      vec[i] = 1;
    }
    std::cout << m_ << std::endl; 
  }
  MatrixXd m_(30, 30);
  SubMatrix m(m_); 
  for(int i=0; i<3; i++){
    SubMatrix a = m.block(10*i, 10*i, 10, 10);
    for(int j=0; j<10; j++){
      SubMatrix c = a.slice(j);
      for(int k=0; k<10; k++){
        c[k] = k;
      }
    }
  }
  std::cout << m_ << std::endl; 
}
