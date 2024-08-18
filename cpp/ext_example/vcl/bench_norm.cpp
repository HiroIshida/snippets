#include "version2/vectorclass.h"
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

int main() {
  size_t n_size = 200;
  Eigen::Matrix3Xd x = Eigen::Matrix3Xd::Random(3, n_size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, n_size-1);

  size_t n = 1000000;
  std::vector<size_t> indices1(n); 
  std::vector<size_t> indices2(n);
  for (size_t i = 0; i < n; i++) {
    indices1[i] = dis(gen);
    indices2[i] = dis(gen);
  }

  {
    // bench eigen
    auto start = std::chrono::high_resolution_clock::now();
    double sqnorm_sum = 0;  // to avoid optimization
    for(int i = 0; i < n; i++) {
      size_t j1 = indices1[i];
      size_t j2 = indices2[i];
      double sqnorm = (x.col(j1) - x.col(j2)).squaredNorm();
      sqnorm_sum += sqnorm;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "eigen: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
    std::cout << "(to avoid optimization) sqnorm_sum: " << sqnorm_sum << std::endl;
  }
  {
    // bench simd using Vec4d
    Eigen::VectorXd xs = x.row(0);
    Eigen::VectorXd ys = x.row(1);
    Eigen::VectorXd zs = x.row(2);
    auto start = std::chrono::high_resolution_clock::now();
    double sqnorm_sum = 0;  // to avoid optimization
    size_t head = 0;
    while(true) { 
      size_t f1, f2, f3, f4;
      size_t s1, s2, s3, s4;
      f1 = indices1[head];
      f2 = indices1[head+1];
      f3 = indices1[head+2];
      f4 = indices1[head+3];
      s1 = indices2[head];
      s2 = indices2[head+1];
      s3 = indices2[head+2];
      s4 = indices2[head+3];
      head += 4;
      if (head >= n) break;
      Vec4d x1(xs[f1], xs[f2], xs[f3], xs[f4]);
      Vec4d y1(ys[f1], ys[f2], ys[f3], ys[f4]);
      Vec4d z1(zs[f1], zs[f2], zs[f3], zs[f4]);
      Vec4d x2(xs[s1], xs[s2], xs[s3], xs[s4]);
      Vec4d y2(ys[s1], ys[s2], ys[s3], ys[s4]);
      Vec4d z2(zs[s1], zs[s2], zs[s3], zs[s4]);
      Vec4d sqnorm = square(x1 - x2) + square(y1 - y2) + square(z1 - z2);
      sqnorm_sum += horizontal_add(sqnorm);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "simd (xs, ys, zs): " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
    std::cout << "(to avoid optimization) sqnorm_sum: " << sqnorm_sum << std::endl;
  }
  {
    // bench simd using Vec4d
    auto start = std::chrono::high_resolution_clock::now();
    double sqnorm_sum = 0;  // to avoid optimization
    size_t head = 0;
    while(true) { 
      size_t f1, f2, f3, f4;
      size_t s1, s2, s3, s4;
      f1 = indices1[head];
      f2 = indices1[head+1];
      f3 = indices1[head+2];
      f4 = indices1[head+3];
      s1 = indices2[head];
      s2 = indices2[head+1];
      s3 = indices2[head+2];
      s4 = indices2[head+3];
      head += 4;
      if (head >= n) break;
      Vec4d x1(x.coeff(0, f1), x.coeff(0, f2), x.coeff(0, f3), x.coeff(0, f4));
      Vec4d y1(x.coeff(1, f1), x.coeff(1, f2), x.coeff(1, f3), x.coeff(1, f4));
      Vec4d z1(x.coeff(2, f1), x.coeff(2, f2), x.coeff(2, f3), x.coeff(2, f4));
      Vec4d x2(x.coeff(0, s1), x.coeff(0, s2), x.coeff(0, s3), x.coeff(0, s4));
      Vec4d y2(x.coeff(1, s1), x.coeff(1, s2), x.coeff(1, s3), x.coeff(1, s4));
      Vec4d z2(x.coeff(2, s1), x.coeff(2, s2), x.coeff(2, s3), x.coeff(2, s4));
      Vec4d sqnorm = square(x1 - x2) + square(y1 - y2) + square(z1 - z2);
      sqnorm_sum += horizontal_add(sqnorm);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "simd (coeff): " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
    std::cout << "(to avoid optimization) sqnorm_sum: " << sqnorm_sum << std::endl;
  }
}
