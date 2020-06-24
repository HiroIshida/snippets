#include <Python.h>

/*
static PyObject *hello_internal(PyObject* self) {
   return Py_BuildValue("s", "Hello, Python extensions!!");
}

static PyMethodDef HelloMethods[] = {
    {"hello", hello_internal, METH_VARARGS, "Python interface for fputs C library function"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef hello_module = {
    PyModuleDef_HEAD_INIT,
    "helloworld",
    "Python interface for the fputs C library function",
    -1,
    HelloMethods
};

PyMODINIT_FUNC PyInit_hello(void) {
    return PyModule_Create(&hello_module);
}
*/

static PyObject *method_fputs(PyObject *self, PyObject *args) {

    char *str, *filename = NULL;

    int bytes_copied = -1;


    /* Parse arguments */

    if(!PyArg_ParseTuple(args, "ss", &str, &filename)) {

        return NULL;

    }


    FILE *fp = fopen(filename, "w");

    bytes_copied = fputs(str, fp);

    fclose(fp);


    return PyLong_FromLong(bytes_copied);

}

static PyMethodDef FputsMethods[] = {
    {"fputs", method_fputs, METH_VARARGS, "Python interface for fputs C library function"},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef fputsmodule = {
    PyModuleDef_HEAD_INIT,
    "fputs",
    "Python interface for the fputs C library function",
    -1,
    FputsMethods
};

PyMODINIT_FUNC PyInit_fputs(void) {
    return PyModule_Create(&fputsmodule);
}
