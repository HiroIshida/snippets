#include<memory>
#include<vector>
#include<iostream>

using namespace std;
int main(){
    std::shared_ptr<int> ptr(new int(10));
    cout << ptr.use_count() << endl;
    
    std::shared_ptr<int> hoge(ptr);
    cout << ptr.use_count() << endl;

    std::vector<shared_ptr<int>> vec;
    vec.push_back(hoge);
    cout << ptr.use_count() << endl;

    // reference does not increase reference count!
    auto& geho = hoge;
    cout << ptr.use_count() << endl;
}
