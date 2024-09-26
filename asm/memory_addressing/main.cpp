#include <iostream>
#include <cstdint>

extern "C" int32_t access_rdata(int32_t idx);
extern "C" int64_t access_heap(int64_t* arr, int64_t idx);

int main() {
  // access rdata section
  for(size_t i = 0; i < 10; i++) {
    auto ret = access_rdata(i);
    std::cout << "index: " << i << " value: " << ret << std::endl;
  }

  int64_t *heap = new int64_t[100];
  for(size_t i = 0; i < 100; i++) {
    heap[i] = i;
  }
  for(size_t i = 0; i < 10; i++) {
    auto ret = access_heap(heap, i);
    std::cout << "index: " << i << " value: " << ret << std::endl;
  }
}
