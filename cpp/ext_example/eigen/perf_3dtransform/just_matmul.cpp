#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

using namespace Eigen;
using namespace std;
using namespace std::chrono;

void benchmark_matrix_multiplication() {
    const int num_matrices = 1000000;
    vector<Matrix3d> matrices(num_matrices);
    for(size_t i = 0; i < num_matrices; i++) {
      auto q = Eigen::Quaterniond::UnitRandom();
      matrices[i] = q.toRotationMatrix();
    }

    vector<Matrix4d> matrices4(num_matrices);
    for(size_t i = 0; i < num_matrices; i++) {
      matrices4[i].block<3,3>(0,0) = matrices[i];
    }

    {
      auto start = high_resolution_clock::now();
      Matrix3d result = Matrix3d::Identity();
      for(size_t i = 0; i < num_matrices; i++) {
        asm("# 3dmat mult start");
        result = result * matrices[i];
        asm("# 3dmat mult end");
      }
      auto end = high_resolution_clock::now();
      auto duration = duration_cast<microseconds>(end - start);
      std::cout << "Matrix3d: " << duration.count() << " microseconds" << std::endl;
      std::cout << result << std::endl;
    }

    {
      auto start = high_resolution_clock::now();
      Matrix4d result = Matrix4d::Identity();
      for(size_t i = 0; i < num_matrices; i++) {
        asm("# 4dmat mult start");
        result = result * matrices4[i];
        asm("# 4dmat mult end");
      }
      auto end = high_resolution_clock::now();
      auto duration = duration_cast<microseconds>(end - start);
      std::cout << "Matrix4d: " << duration.count() << " microseconds" << std::endl;
      std::cout << result << std::endl;
    }
}

int main() {
    benchmark_matrix_multiplication();
    return 0;
}
