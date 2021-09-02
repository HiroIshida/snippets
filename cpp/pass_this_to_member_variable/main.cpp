class A;
struct B
{
  A* ptr;
};

class A
{
  public:
    A() : b(B{this}) {};
    B b;
};

int main(){
  auto a = A();
}
