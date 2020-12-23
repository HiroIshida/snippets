#include <stdio.h>
#include <Python.h>
#include <string>

using namespace std;

int main()
{
  PyObject* pInt;
  Py_Initialize();
  string site_package_path = "/home/hiro/.local/lib/python2.7/site-packages";
  PySys_SetPath("/home/hiro/.local/lib/python2.7/site-packages");
  auto module = PyImport_ImportModule("tinyfk");
  PyRun_SimpleString("print('Hello World from Embedded Python!!!')");
  Py_Finalize();
  return 0;
}
