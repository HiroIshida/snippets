#include<vector>
#include<iostream>

using namespace std;

int main(){
  int bench_itr = 1000000;
  int n = 200;
  auto vecvec = vector<vector<bool>>(n, vector<bool>(n));
  {// access by []
    clock_t start = clock();
    bool tmp;
    for(int i=0; i<bench_itr; i++){
      for(int j=0; j<n; j++){
        for(int k=0; k<n; k++){
          tmp = tmp || vecvec[j][k]; // to avoid compiler optimization
        }
      }
    }
    clock_t end = clock();
    std::cout << end - start << std::endl; 
    std::cout << tmp << std::endl; // must print this to avoid compiler optimization
  }
  {// access by at()
    clock_t start = clock();
    bool tmp;
    for(int i=0; i<bench_itr; i++){
      for(int j=0; j<n; j++){
        for(int k=0; k<n; k++){
          tmp = tmp || vecvec.at(j).at(k);
        }
      }
    }
    clock_t end = clock();
    std::cout << end - start << std::endl; 
    std::cout << tmp << std::endl; 
  }
/* output
66101
1
89432
1
*/
}
