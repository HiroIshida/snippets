#include <iostream>

struct Rotation
{
  Rotation(double _w, double _x, double _y, double _z) : w(_w), x(_x), y(_y), z(_z) {}
  Rotation() : w(0), x(0), y(0), z(0) {}
  double w;
  double x;
  double y;
  double z;

  Rotation operator*( const Rotation &qt ) const{
      Rotation c;
      c.x = this->w * qt.x + this->x * qt.w + this->y * qt.z - this->z * qt.y;
      c.y = this->w * qt.y - this->x * qt.z + this->y * qt.w + this->z * qt.x;
      c.z = this->w * qt.z + this->x * qt.y - this->y * qt.x + this->z * qt.w;
      c.w = this->w * qt.w - this->x * qt.x - this->y * qt.y - this->z * qt.z;
      return c;
  }
};

int main(){
    Rotation r1(1, 0, 0, 0);
    Rotation r2(0, 1, 0, 0);
    auto a = r1 * r2;
    std::cout << a.x << std::endl;
}
