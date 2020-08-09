#include<iostream>
#include<map>
/*
 *  mutable keyward
 *  use at() for map function when the map is const
 */
class Sample
{
  public:
    Sample(int var_) : var(var_){
      for(int i=0; i<10; i++){
        table[i] = 2 * i;
      }
    }
    mutable std::map<int, int> cache;
    std::map<int, int> table;
    int var;

    int access(int x) const
    {
      //int hoge = table[var]; compile error
      int hoge = table.at(var);
    }

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
