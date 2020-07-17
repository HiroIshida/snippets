#include <Python.h>

static PyObject *method_fputs(PyObject *self, PyObject *args) {
    char *str, *filename = NULL;
    PyObject* pymat;
    PyArg_ParseTuple(args, "O", &pymat);

    double mat[3][3];
    for(int i=0; i<3; i++){
        PyObject* pyvec = PyTuple_GetItem(pymat, i);
        for(int j=0; j<3; j++){
            mat[i][j] = PyFloat_AsDouble(PyTuple_GetItem(pyvec, j));
        }
    }

    PyObject* pymat_ret = PyTuple_New(3);
    for(int i=0; i<3; i++){
        PyObject* pyvec_ret = PyTuple_New(3);
        for(int j=0; j<3; j++){
            PyTuple_SetItem(pyvec_ret, j, PyFloat_FromDouble(mat[i][j]));
        }
        PyTuple_SetItem(pymat_ret, i, pyvec_ret);
    }
	return pymat_ret;
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
