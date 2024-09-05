#include <cstddef>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cstdint>
#include <bitset>

bool is64ByteAligned(const void* ptr) {
    std::uintptr_t address = reinterpret_cast<std::uintptr_t>(ptr);
    // the first 6 bits should be zero
    return (address & 0x3F) == 0;
}

struct QuatTrans {
  Eigen::Quaterniond q;
  Eigen::Vector3d t;
};

struct CustomQuatTrans {
  double qx, qy, qz, qw;
  double tx, ty, tz;
};

template <typename T>
void test(){
  T *qt = new T[10000];
  std::cout << "unalined => " << std::endl;
  std::cout << qt << std::endl;
  std::cout << "is 64byte aligned? " << is64ByteAligned(qt) << std::endl;
  auto dist = reinterpret_cast<std::uintptr_t>(&qt[1]) - reinterpret_cast<std::uintptr_t>(&qt[0]);
  std::cout << "distance between two consecutive elements: " << dist << std::endl;

  T *qt_aligned = static_cast<T*>(std::aligned_alloc(64, 10000 * sizeof(T)));
  std::cout << "aligned => " << std::endl;
  std::cout << qt_aligned << std::endl;
  std::cout << "is 64byte aligned? " << is64ByteAligned(qt_aligned) << std::endl;
  auto dist_aligned = reinterpret_cast<std::uintptr_t>(&qt_aligned[1]) - reinterpret_cast<std::uintptr_t>(&qt_aligned[0]);
  std::cout << "distance between two consecutive elements: " << dist_aligned << std::endl;
}

int main(){
  std::cout << "testing Custom Quat Trans" << std::endl;
  // aligned_alloc does not care adjacent elements are 64byte aligned or not
  test<CustomQuatTrans>();

  std::cout << "testing Eigen Quat Trans" << std::endl;
  // Eigen internally pads the struct to make sure adjacent elements are 64byte aligned
  // but it is not the effect of aligned_alloc
  test<QuatTrans>();
}

