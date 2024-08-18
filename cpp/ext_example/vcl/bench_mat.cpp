#include "version2/vectorclass.h"
#include <Eigen/Dense>
#include <iostream>
#include <chrono>

using namespace Eigen;

Eigen::Matrix4d multiply_simd(const Eigen::Matrix4d &A, const Eigen::Matrix4d &B) {
  Vec4d a_row0(A(0,0), A(0,1), A(0,2), A(0,3));
  Vec4d a_row1(A(1,0), A(1,1), A(1,2), A(1,3));
  Vec4d a_row2(A(2,0), A(2,1), A(2,2), A(2,3));
  Vec4d a_row3(A(3,0), A(3,1), A(3,2), A(3,3));

  Vec4d b_col0(B(0,0), B(1,0), B(2,0), B(3,0));
  Vec4d b_col1(B(0,1), B(1,1), B(2,1), B(3,1));
  Vec4d b_col2(B(0,2), B(1,2), B(2,2), B(3,2));
  Vec4d b_col3(B(0,3), B(1,3), B(2,3), B(3,3));

  Eigen::Matrix4d C;
  C << horizontal_add(a_row0 * b_col0), horizontal_add(a_row0 * b_col1), horizontal_add(a_row0 * b_col2), horizontal_add(a_row0 * b_col3),
       horizontal_add(a_row1 * b_col0), horizontal_add(a_row1 * b_col1), horizontal_add(a_row1 * b_col2), horizontal_add(a_row1 * b_col3),
       horizontal_add(a_row2 * b_col0), horizontal_add(a_row2 * b_col1), horizontal_add(a_row2 * b_col2), horizontal_add(a_row2 * b_col3),
       horizontal_add(a_row3 * b_col0), horizontal_add(a_row3 * b_col1), horizontal_add(a_row3 * b_col2), horizontal_add(a_row3 * b_col3);
  return C;
}

int main() {
  Eigen::Matrix4d A, B;
  A << 1, 0, 0, 0,
       0, 1, 1, 0,
       0, 0, 1, 0,
       0, 0, 0, 1;
  B << 1, 0, 0, 1,
       0, 1, 0, 0,
       0, 0, 1, 0,
       0, 0, 0, 1;
  {
    auto A_copied = A;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000000; i++) {
      A_copied = A_copied * B;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Eigen: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    std::cout << "avoid compiler optimization: " << A_copied(0,0) << std::endl;
  }

  {
    auto A_copied = A;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000000; i++) {
      A_copied = multiply_simd(A_copied, B);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "SIMD: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    std::cout << "avoid compiler optimization: " << A_copied(0,0) << std::endl;
  }
}


