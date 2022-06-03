#include <iostream>
#include <chrono>
#include <Eigen/Core>

using namespace std::chrono;

struct Quaternion
{
  double x;
  double y;
  double z;
  double w;

  Quaternion operator*(const Quaternion & q){
    Quaternion c;
    c.x = this->w * q.x + this->x * q.w + this->y * q.z - this->z * q.y;
    c.y = this->w * q.y - this->x * q.z + this->y * q.w + this->z * q.x;
    c.z = this->w * q.z + this->x * q.y - this->y * q.x + this->z * q.w;
    c.w = this->w * q.w - this->x * q.x - this->y * q.y - this->z * q.z;
    return c;
  }
};

int main(){
  int N = 10000000;
  {
    // quaternion
    Quaternion q{0.1, 0, 0, 0.9};
    Quaternion q_out = q;

    auto start = high_resolution_clock::now();
    for (int i = 0; i < N; i++){
      q_out = q_out * q;
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    std::cout << duration.count() << std::endl;
    std::cout << q_out.w << std::endl;
  }

  {
    Eigen::Matrix3d m;
    Eigen::Matrix3d m_out;

    auto start = high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
      m_out *= m;
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    std::cout << duration.count() << std::endl;
    std::cout << m_out << std::endl;
  }

}
