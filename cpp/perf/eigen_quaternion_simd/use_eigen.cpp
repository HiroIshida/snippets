#include <iostream>
#include <Eigen/Geometry>

class Rotation
{
public:
    Eigen::Quaterniond q;

    Rotation(double w, double x, double y, double z) : q(w, x, y, z) {}

    Rotation operator*(const Rotation &qt) const
    {
        Eigen::Quaterniond result = q * qt.q;
        return Rotation(result.w(), result.x(), result.y(), result.z());
    }
};

int main()
{
    Rotation r1(1, 0, 0, 0);
    Rotation r2(0, 1, 0, 0);
    auto a = r1 * r2;
    std::cout << a.q.x() << std::endl;
    // std::cout << a.q << std::endl;
}
