# g++  $(llvm-config --cxxflags --ldflags --system-libs --libs all) -DLLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1 make_ir.cpp
clang++  $(llvm-config --cxxflags --ldflags --libs core) -DLLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1 make_ir.cpp
