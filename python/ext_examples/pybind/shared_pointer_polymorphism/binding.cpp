#include <pybind11/pybind11.h>

namespace py = pybind11;

struct Base { };
struct Derived : public Base { };

struct Container { void input(const std::shared_ptr<Base>& ptr) { } };

PYBIND11_MODULE(tmp, m)
{
    py::class_<Base,    std::shared_ptr<Base>>(m, "Base").def(py::init<>());
    // py::class_<Derived, std::shared_ptr<Derived>>(m, "Derived").def(py::init<>()); // this doesn't work
    py::class_<Derived, std::shared_ptr<Derived>, Base>(m, "Derived").def(py::init<>());  // we need tell pybind11 that Derived is derived from Base

    py::class_<Container, std::shared_ptr<Container>>(m, "Container")
        .def(py::init<>())
        .def("input", &Container::input);
}
