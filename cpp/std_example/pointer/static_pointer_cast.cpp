#include <string>
#include <iostream>
#include <memory>

class Geometry
{
public:
  enum {SPHERE, BOX, CYLINDER, MESH} type;
};

class Mesh : public Geometry
{
public:
  Mesh() {type = MESH;};
  std::string filename;
};

class Sphere: public Geometry
{
  public:
    Sphere() {type = SPHERE;};
    double r;
};

int main(){
  std::shared_ptr<Geometry> geom;
  Mesh *m = new Mesh();
  m->filename = "ishida";
  geom.reset(m);
  //geom->filename; // this causes error at compilation
  
  /*
   * if Geometry class has virtual function, dynamic cast is possible.
   *
   * I first thought that this kinf of implementation is a kind of design error
   * but actually seems to be correct usage. see:
   * http://yohshiy.blog.fc2.com/blog-entry-15.html
   */
  auto m_ = std::static_pointer_cast<Mesh>(geom);
  std::cout << m_->filename << std::endl;
}
