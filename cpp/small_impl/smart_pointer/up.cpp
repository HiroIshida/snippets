#include <iostream>

template <typename T>
struct UP{
  T* ptr_;
  UP() : ptr_(nullptr) {}
  UP(T* ptr) : ptr_(ptr) {}
  UP(const UP& other) = delete;
  UP& operator=(const UP& other) = delete;
  T* operator->() const {return ptr_;}

  UP(UP&& other) : ptr_(nullptr) {
    ptr_ = other.ptr_;
    other.ptr_ = nullptr;
  }
  UP& operator=(UP&& other) {
    if(ptr_) {
      delete ptr_;
    }
    ptr_ = other.ptr_;
    other.ptr_ = nullptr;
    return *this;
  }

  ~UP(){
    if(ptr_){
      std::cout << "release" << std::endl;
    }
    delete ptr_;
  }
};

int main() {
  {
    auto a = UP<double>(new double(1.0));
    {
      auto b = UP<double>(new double(1.0));
      {
        auto c = std::move(b);
      }
    }
  }
}
