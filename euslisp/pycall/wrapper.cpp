#include <Python.h>
#include <iostream>
#include <string>

using namespace std;

void*  __PyImport_ImportModule(const char* module_name){
  Py_Initialize();
  PyObject* module = PyImport_ImportModule(module_name);
  char* name = PyModule_GetName(module);
  std::cout << "[pywrap] imported : " << name << std::endl; 
  char* filepath = PyModule_GetFilename(module);
  std::cout << "[pywrap] import path : " << filepath << std::endl; 
  return (void*)module;
}

long __PyObject_CallMethod(void* module_, const char* module_method){
  Py_Initialize();
  auto module = (PyObject*)module_;
  PyObject_CallMethod(module, const_cast<char*>(module_method), NULL);
  return 0;
}

extern "C" {
    void* _PyImport_ImportModule(const char* module_name){
        return __PyImport_ImportModule(module_name);
    }
    long _PyObject_CallMethod(void* module_, const char* module_method){
        return __PyObject_CallMethod(module_, module_method);
    }
}
