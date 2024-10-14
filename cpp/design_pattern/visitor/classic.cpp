#include <iostream>
#include <memory>
#include <vector>

class Sphere;
class Box;

class PrimitiveVisitor {
  public:
    ~PrimitiveVisitor() = default;
    virtual void visit(Sphere const& sphere) const = 0;
    virtual void visit(Box const& box) const = 0;
};

class Primitive {  
  public:
    using Ptr = std::unique_ptr<Primitive>;
    Primitive() = default;
    virtual ~Primitive() = default;
    virtual void accept(PrimitiveVisitor const& visitor) = 0;
};

class Sphere : public Primitive {
  public:
    using Ptr = std::unique_ptr<Sphere>;
    Sphere() = default;
    // this is required cause *this has different type in each derived class
    void accept(PrimitiveVisitor const& visitor) override { visitor.visit(*this); }
};

class Box : public Primitive {
  public:
    using Ptr = std::unique_ptr<Box>;
    Box() = default;
    // this is required cause *this has different type in each derived class
    void accept(PrimitiveVisitor const& visitor) override { visitor.visit(*this); }
};

class PrintVisitor : public PrimitiveVisitor {
  public:
    void visit(Sphere const& sphere) const override { std::cout << "Sphere" << std::endl; }
    void visit(Box const& box) const override { std::cout << "Box" << std::endl; }
};

int main() {
  auto printer = PrintVisitor();
  std::vector<Primitive::Ptr> vec;
  vec.push_back(std::make_unique<Sphere>());
  vec.push_back(std::make_unique<Box>());
  for(auto const& p : vec) {
    p->accept(printer);
  }
}
