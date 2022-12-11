#include <Python.h>
#include <iostream>
#include <string>

using namespace std;

long  __PyImport_ImportModule(const char* module_name){
  Py_Initialize();
  PyObject* module = PyImport_ImportModule(module_name);
  char* name = PyModule_GetName(module);
  std::cout << "[pywrap] imported : " << name << std::endl; 
  char* filepath = PyModule_GetFilename(module);
  std::cout << "[pywrap] import path : " << filepath << std::endl; 
  std::cout << (long)((void*)module) << std::endl;
  return (long)((void*)module);
}

long __PyObject_CallMethod(long module_, const char* module_method){
  // Py_Initialize();
  auto module = (PyObject*)((void*)module_);
  std::cout << module_method << std::endl;
  PyObject_CallMethod(module, "exit", NULL);
  return -1;
}

extern "C" {
    long _PyImport_ImportModule(const char* module_name){
        return __PyImport_ImportModule(module_name);
    }
    long _PyObject_CallMethod(long module_, const char* module_method){
        return __PyObject_CallMethod(module_, module_method);
    }
}
