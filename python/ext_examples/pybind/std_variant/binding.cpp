#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <variant>
#include <iostream>

namespace py = pybind11;

struct Sphere { double r; };
struct Box { double x, y, z; };
using Shape = std::variant<Sphere, Box>;

void func(const Shape& shape)
{
  // use get_if to access the value of the variant
  if (auto sphere = std::get_if<Sphere>(&shape))
  {
    std::cout << "Sphere: " << sphere->r << std::endl;
  }
  else if (auto box = std::get_if<Box>(&shape))
  {
    std::cout << "Box: " << box->x << ", " << box->y << ", " << box->z << std::endl;
  }
}

void loop(const std::vector<Shape>& shapes)
{
  for (const auto& shape : shapes)
  {
    func(shape);
  }
}

PYBIND11_MODULE(tmp, m)
{
  py::class_<Sphere>(m, "Sphere")
    .def(py::init<double>());
  py::class_<Box>(m, "Box")
    .def(py::init<double, double, double>());
  m.def("func", &func);
  m.def("loop", &loop);
}
