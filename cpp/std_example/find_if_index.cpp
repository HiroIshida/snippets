#include <iterator>
#include<vector>
#include<iostream>
#include<algorithm>
using namespace std;

int main(){
  std::vector<bool> a(4);
  a[3] = true;
  for(auto b: a){
    std::cout << b << std::endl; 
  }
  // NOTE THAT find returns value, not index
  auto it = std::find(a.begin(), a.end(), true);
  const size_t index = std::distance(a.begin(), it);
  std::cout << index << std::endl; 
}
