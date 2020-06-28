#include<iostream>
using namespace std;
class A{
    public:
        void say(){cout<< "Im A"<< endl;}
        int data;
};
class B: public A{
    public:
        void say(){cout<< "Im B"<< endl;}
};

class C{
    A a;
    public:
        C(B b): a(b) {}
        void say(){a.say();}
};
int main()
{
    auto b = B();
    b.say();
    auto c = C(b);
    c.say();
}
