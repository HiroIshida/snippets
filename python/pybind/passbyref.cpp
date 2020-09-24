#include "pybind11/pybind11.h"
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>

using namespace Eigen;
namespace py = pybind11;
// https://pybind11.readthedocs.io/en/master/advanced/cast/eigen.html#pass-by-reference

void fill_by_zero_ref(py::EigenDRef<VectorXd> v)
{
    v *= 0.0;
}

VectorXd fill_by_zero_val(const VectorXd& hoge)
{
  VectorXd hage = hoge;
  hage *= 0.0;
  return hage;
}

PYBIND11_MODULE(example, m) {
    m.def("fbz_ref", &fill_by_zero_ref, "hoge");
    m.def("fbz_val", &fill_by_zero_val, "hoge");
}
