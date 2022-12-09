#include <Python.h>
#include <iostream>
#include <string>

using namespace std;

void initialize_python_ifnotyet()
{
  const int is_init = Py_IsInitialized();
  if(is_init == 0){
    Py_Initialize();
  }
}


void*  __PyImport_ImportModule(const char* module_name){
  initialize_python_ifnotyet();
  PyObject* module = PyImport_ImportModule(module_name);
  return (void*)module;
}

int main(){
  const auto module = __PyImport_ImportModule("os");
}
