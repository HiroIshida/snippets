#include<iostream>
#include<map>
class Sample
{
  public:
    Sample(int var_) : var(var_){}
    mutable std::map<int, int> cache;
    int var;
    int heavy_computation(int x) const
    {
      std::map<int, int>::iterator iter = cache.find(x);
      if(iter == cache.end()){ // cache doesn't exist
        int y = var + x; 
        cache[x] =y;
        return var;
      }
      std::cout << "cached one" << std::endl; 
      return iter->second;
    }
};

int main(){
  Sample s(0);
  s.heavy_computation(1);
  s.heavy_computation(1);
}
