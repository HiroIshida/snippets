#include <iostream>
#include <cstdint>

extern "C" int32_t access_rdata(int32_t idx);

int main() {
  for(size_t i = 0; i < 10; i++) {
    auto ret = access_rdata(i);
    std::cout << "index: " << i << " value: " << ret << std::endl;
  }
}
