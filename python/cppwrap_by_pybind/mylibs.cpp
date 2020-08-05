#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // vector用
#include "mylibs.h"

int add(int x, int y) {
    return x+y;
}

vector<int> vec_double(vector<int> &vec) {
    for(auto &v : vec) {
        v *= 2;
    }
    return vec;
}

vector<vector<int>> vec_add(vector<vector<int>> &vec) {
    vector<vector<int>> result(vec.size(), vector<int>());
    for(int i = 0; i < vec.size(); i++) {
        int tmp = 0;
        for(auto &t : vec[i]) {
            tmp += t;
            result[i].push_back(tmp);
        }
    }
    return result;
}

POINT move_p(POINT p, int d) {
    return POINT(p.X() + d, p.Y() + d);
}

namespace py = pybind11;
PYBIND11_PLUGIN(mylibs) {
    py::module m("mylibs", "mylibs made by pybind11");
    m.def("add", &add);
    m.def("vec_double", &vec_double);
    m.def("vec_add", &vec_add);

    py::class_<POINT>(m, "POINT")
        .def_readwrite("sum", &POINT::sum)
        .def(py::init<int, int>())
        .def("X", &POINT::X)
        .def("Y", &POINT::Y);
    m.def("move_p", &move_p);

    return m.ptr();
}
