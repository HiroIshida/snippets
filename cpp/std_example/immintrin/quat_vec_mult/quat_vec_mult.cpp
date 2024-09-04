#include <iostream>
#include <chrono>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <random>
#include <immintrin.h>
#include <fstream>

#ifdef __AVX512F__
  const size_t bytes_alligned = 64;
  const size_t n_batch = 8;
#define SIMD_LOAD _mm512_load_pd
#define SIMD_STORE _mm512_store_pd
#define SIMD_ADD _mm512_add_pd
#define SIMD_SUB _mm512_sub_pd
#define SIMD_MUL _mm512_mul_pd
#define SIMD_SET1 _mm512_set1_pd

#elif __AVX2__ 
  const size_t bytes_alligned = 32;
  const size_t n_batch = 4;
#define SIMD_LOAD _mm256_load_pd
#define SIMD_STORE _mm256_store_pd
#define SIMD_ADD _mm256_add_pd
#define SIMD_SUB _mm256_sub_pd
#define SIMD_MUL _mm256_mul_pd
#define SIMD_SET1 _mm256_set1_pd

#else
  const size_t bytes_alligned = 16;
  const size_t n_batch = 2;
#define SIMD_LOAD _mm_load_pd
#define SIMD_STORE _mm_store_pd
#define SIMD_ADD _mm_add_pd
#define SIMD_SUB _mm_sub_pd
#define SIMD_MUL _mm_mul_pd
#define SIMD_SET1 _mm_set1_pd
#endif

void quat_mult_vec(
    double* qx, double* qy, double* qz, double* qw,
    double* x, double* y, double* z,
    double* output_x, double* output_y, double* output_z){
  auto qx_vec = SIMD_LOAD(qx);
  auto qy_vec = SIMD_LOAD(qy);
  auto qz_vec = SIMD_LOAD(qz);
  auto qw_vec = SIMD_LOAD(qw);
  auto x_vec = SIMD_LOAD(x);
  auto y_vec = SIMD_LOAD(y);
  auto z_vec = SIMD_LOAD(z);
  auto ones = SIMD_SET1(1);
  auto twos = SIMD_SET1(2);

  auto qxqx_2 = SIMD_MUL(SIMD_MUL(qx_vec, qx_vec), twos);
  auto qyqy_2 = SIMD_MUL(SIMD_MUL(qy_vec, qy_vec), twos);
  auto qzqz_2 = SIMD_MUL(SIMD_MUL(qz_vec, qz_vec), twos);
  auto qwqw_2 = SIMD_MUL(SIMD_MUL(qw_vec, qw_vec), twos);
  auto qxqy_2 = SIMD_MUL(SIMD_MUL(qx_vec, qy_vec), twos);
  auto qxqz_2 = SIMD_MUL(SIMD_MUL(qx_vec, qz_vec), twos);
  auto qxqw_2 = SIMD_MUL(SIMD_MUL(qx_vec, qw_vec), twos);
  auto qyqz_2 = SIMD_MUL(SIMD_MUL(qy_vec, qz_vec), twos);
  auto qyqw_2 = SIMD_MUL(SIMD_MUL(qy_vec, qw_vec), twos);
  auto qzqw_2 = SIMD_MUL(SIMD_MUL(qz_vec, qw_vec), twos);

  auto M00 = SIMD_SUB(SIMD_ADD(qwqw_2, qxqx_2), ones);
  auto M01 = SIMD_SUB(qxqy_2, qzqw_2);
  auto M02 = SIMD_ADD(qxqz_2, qyqw_2);
  auto output_x_vec = SIMD_ADD(SIMD_ADD(SIMD_MUL(M00, x_vec), SIMD_MUL(M01, y_vec)), SIMD_MUL(M02, z_vec));
  SIMD_STORE(output_x, output_x_vec);

  auto M10 = SIMD_ADD(qxqy_2, qzqw_2);
  auto M11 = SIMD_SUB(SIMD_ADD(qwqw_2, qyqy_2), ones);
  auto M12 = SIMD_SUB(qyqz_2, qxqw_2);
  auto output_y_vec = SIMD_ADD(SIMD_ADD(SIMD_MUL(M10, x_vec), SIMD_MUL(M11, y_vec)), SIMD_MUL(M12, z_vec));
  SIMD_STORE(output_y, output_y_vec);

  auto M20 = SIMD_SUB(qxqz_2, qyqw_2);
  auto M21 = SIMD_ADD(qyqz_2, qxqw_2);
  auto M22 = SIMD_SUB(SIMD_ADD(qwqw_2, qzqz_2), ones);
  auto output_z_vec = SIMD_ADD(SIMD_ADD(SIMD_MUL(M20, x_vec), SIMD_MUL(M21, y_vec)), SIMD_MUL(M22, z_vec));
  SIMD_STORE(output_z, output_z_vec);
}


int main(){
  size_t N = 40;
  size_t M = 100000000;
  size_t aligned_N = N + (n_batch - N % n_batch);

  double* qx = static_cast<double*>(aligned_alloc(bytes_alligned, aligned_N * sizeof(double)));
  double* qy = static_cast<double*>(aligned_alloc(bytes_alligned, aligned_N * sizeof(double)));
  double* qz = static_cast<double*>(aligned_alloc(bytes_alligned, aligned_N * sizeof(double)));
  double* qw = static_cast<double*>(aligned_alloc(bytes_alligned, aligned_N * sizeof(double)));
  double* x = static_cast<double*>(aligned_alloc(bytes_alligned, aligned_N * sizeof(double)));
  double* y = static_cast<double*>(aligned_alloc(bytes_alligned, aligned_N * sizeof(double)));
  double* z = static_cast<double*>(aligned_alloc(bytes_alligned, aligned_N * sizeof(double)));
  double* output_x = static_cast<double*>(aligned_alloc(bytes_alligned, aligned_N * sizeof(double)));
  double* output_y = static_cast<double*>(aligned_alloc(bytes_alligned, aligned_N * sizeof(double)));
  double* output_z = static_cast<double*>(aligned_alloc(bytes_alligned, aligned_N * sizeof(double)));

  for(size_t i = 0; i < N; ++i) {
    Eigen::Vector4d q = Eigen::Vector4d::Random();
    Eigen::Vector3d v = Eigen::Vector3d::Random();
    qx[i] = q[0]; qy[i] = q[1]; qz[i] = q[2]; qw[i] = q[3];
    x[i] = v[0]; y[i] = v[1]; z[i] = v[2];
  }


  int number;
  std::cout << "Enter a number: ";
  std::cin >> number;  // Read a number from the user
  std::cout << "start" << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  for(size_t i = 0; i < M; ++i){
    for(size_t head = 0; head < N; head += n_batch){
      quat_mult_vec(qx + head, qy + head, qz + head, qw + head, x + head, y + head, z + head, output_x + head, output_y + head, output_z + head);
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;

  double sum = 0;
  for(size_t i = 0; i < N; ++i) {
    sum += output_x[i] + output_y[i] + output_z[i];
  }
  std::cout << "Sum: " << sum << std::endl;
}
