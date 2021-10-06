// https://gist.github.com/tonyarkles/e702e4d5b530a9b7cd3fc9837c6609f2
#include <vector>
#include <stdio.h>

int main(int argc, char* argv[]) {
  std::vector<int> foo(10, 0);
  std::vector<int> bar(10, 1);
  for(int i = 0; i < 20; i++) {
    foo[i] = 42;
  }
  bar.clear(); // causes munmap_chunk(): invalid pointer
}
