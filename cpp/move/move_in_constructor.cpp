#include<iostream>
#include<vector>
#include <time.h>
using namespace std;

class TestKuso
{
  public:
    vector<double> heck;
    TestKuso(vector<double> aho){
      heck = aho;
    }
};

class TestRef
{
  public:
    vector<double> heck;
    TestRef(vector<double>& aho){
      heck = aho;
    }
};

class TestRefMove // using move
{
  public:
    vector<double> heck;
    TestRefMove(vector<double> aho) : heck(std::move(aho)) {}
};

int main(){
  int N = 10 * 1000 * 1000;
  vector<double> a(N);
  for(int n; n < N; n++){
  }

  {
    clock_t start = clock();
    auto x = TestKuso(a);
    clock_t end = clock();
    cout << end - start << endl;
  }

  {
    clock_t start = clock();
    auto x = TestRef(a);
    clock_t end = clock();
    cout << end - start << endl;
  }

  {
    clock_t start = clock();
    auto x = TestRefMove(std::move(a));
    clock_t end = clock();
    cout << end - start << endl;
  }
}
