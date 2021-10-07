#include<unordered_map>
#include<iostream>
#include <time.h>

using namespace std;
int main(){
  unordered_map<string, string> a;
  unordered_map<int, string> b;

  a["kusakusanokusanokusa"] = "kusai";

  b[1] = "ishida";

  int N = 20 * 100 * 100;

  {
    clock_t start = clock();
    for(int i=0; i<N; i++){
      a["unko"];
    }
    clock_t end = clock();
    cout << (end - start) * 1e-6 << endl;
  }

  {
    clock_t start = clock();
    for(int i=0; i<N; i++){
      b[1];
    }
    clock_t end = clock();
    cout << (end - start) * 1e-6 << endl;
  }
}
