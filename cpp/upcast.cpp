#include<iostream>
using namespace std;
class A{
    public:
        virtual void say(){cout<< "Im A"<< endl;}
        int data;
};
class B: public A{
    public:
        void say() override {cout<< "Im B"<< endl;}
};

class C{
    A* a;
    public:
        C(A* ptr): a(ptr) {}
        void say(){a->say();}
};
int main()
{
  A* b = new B();
  auto c = C(b);
  c.say();
}
