#include "xbyak/xbyak/xbyak.h"
#include <iostream>
#include <cmath>

struct Codegen : Xbyak::CodeGenerator {
  Codegen(){
    double (*cos_ptr)(double) = std::cos;
    void* cos_vptr = reinterpret_cast<void*>(cos_ptr);
    endbr64();
    push(rbp);
    mov(rbp, rsp);
    sub(rsp, 128);

    vmovsd(xmm0, ptr[rdi]);
    sub(rsp, 8);
    call(cos_vptr);
    add(rsp, 8);
    vmovsd(xmm1, xmm0);
    vmovsd(ptr[rsi], xmm1);

    // mov(rsp, rbp);
    add(rsp, 128);
    pop(rbp);
    ret();
  }
};

int main() {
  Codegen c;
  auto f = c.getCode<void (*)(double*, double*)>();
  double x = 0.0;
  f(&x, &x);
  std::cout << x << std::endl;
  return 0;
}
