#include <iostream>
#include <memory>
#include <stdexcept>

template <typename T>
class SP{
private:
size_t* count_;
T* ptr_;

void process_counter() {
  if(!count_) {
    count_ = new size_t(1);
    std::cout << "init counter" << std::endl;
  }else{
    (*count_)++;
    std::cout << "counter incremented to " << *count_ << std::endl;
  }
}

void release() {
  delete count_;
  delete ptr_;
  std::cout << "release!" << std::endl;
}

public:
SP() : count_(nullptr), ptr_(nullptr) {}
SP(T* ptr) : ptr_(nullptr), count_(nullptr) {
  process_counter();
  ptr_ = ptr;
}
T* operator->() const { return ptr_; }

SP(const SP& other) {
  ptr_ = other.ptr_;
  count_ = other.count_;
  process_counter();
}

SP(SP&& other) : ptr_(other.ptr_), count_(other.count_) {
  other.ptr_ = nullptr;
  other.count_ = nullptr;
}

// SP& not SP to handle the case like a = b = c; and it's standard of C++
SP& operator=(const SP& sp) {
  count_ = sp.count_;
  ptr_ = sp.ptr_;
  process_counter();
  return *this;
}

SP& operator=(SP&& other) {
  count_ = other.count_;
  ptr_ = other.ptr_;
  other.count_ = nullptr;
  other.ptr_ = nullptr;
}

~SP() {
  if(count_) { 
    (*count_)--;
    std::cout << "decl to " << *count_ << std::endl;
    if(*count_ == 0){
      release();
    }
  }
}

};

int main(){
  {
    auto sp = SP<double>(new double(1.0));
    auto sp2 = std::move(sp);
  }
}
