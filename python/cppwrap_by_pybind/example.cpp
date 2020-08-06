#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

using namespace Eigen;
namespace py = pybind11;
struct Pet {
    Pet(const std::string &name) : name(name) {
      std::vector<double> hoge(0);
      vec = hoge;
    }
    void setName(const std::string &name_) { name = name_; }
    const std::string &getName() const { return name; }
    const std::vector<double> &getVec() const { return vec; }
    void pushVec(double val){vec.push_back(val);}
    MatrixXd getMat(){
      MatrixXd m = MatrixXd::Zero(3, 6);
      return m;
    }

    std::string name;
    std::vector<double> vec;
};

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("add", &add, "A function which adds two numbers");
    py::class_<Pet>(m, "Pet")
            .def(py::init<const std::string &>())
            .def("pushvec", &Pet::pushVec)
            .def("getvec", &Pet::getVec)
            .def("setName", &Pet::setName)
            .def("getmat", &Pet::getMat)
            .def("getName", &Pet::getName);
}
