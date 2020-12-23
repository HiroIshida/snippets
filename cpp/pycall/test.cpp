#include <stdio.h>
#include <Python.h>
#include <string>
#include <iostream>

using namespace std;

int main()
{
  PyObject* pInt;
  Py_Initialize();
  string site_package_path = "/home/hiro/.local/lib/python2.7/site-packages";
  PySys_SetPath("/home/hiro/.local/lib/python2.7/site-packages");
  PyObject* module = PyImport_ImportModule("tinyfk");
  char* name = PyModule_GetName(module);
  std::cout << "imported : " << name << std::endl; 
  char* filepath = PyModule_GetFilename(module);
  std::cout << filepath << std::endl; 

  string urdf_file = "/home/hiro/.skrobot/pr2_description/pr2.urdf";
  PyObject* robot_model = PyObject_CallMethod(module, const_cast<char*>("RobotModel"), 
      const_cast<char*>("s"), urdf_file.c_str());

  // https://docs.python.org/ja/3/c-api/call.html#c.PyObject_CallMethod
  string joint_names = "r_elbow_flex_joint";
  // for format https://docs.python.org/3/c-api/arg.html
  // to set list of [string] 
  PyObject* indices = PyObject_CallMethod(robot_model, const_cast<char*>("get_joint_ids"), const_cast<char*>("[s]"), joint_names.c_str());

  PyObject* idx_pyo = PyList_GetItem(indices, 0);
  int idx = PyLong_AsLong(idx_pyo);
  std::cout << "corresponding link index is : " << idx << std::endl; 

  PyRun_SimpleString("print('tinyfk demo')");
  Py_Finalize();
  return 0;
}
