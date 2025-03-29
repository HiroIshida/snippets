#include <Python.h>

// We'll keep a global reference to the callback
static PyObject *global_callback = NULL;

// Function to start an infinite loop that calls the Python callback
static PyObject* start_infinite_loop(PyObject* self, PyObject* args)
{
    PyObject *callback_temp = NULL;

    // Parse arguments: the function expects a single callable object
    if (!PyArg_ParseTuple(args, "O", &callback_temp)) {
        return NULL;
    }

    // Verify that this parameter is in fact a callable
    if (!PyCallable_Check(callback_temp)) {
        PyErr_SetString(PyExc_TypeError, "Parameter must be callable");
        return NULL;
    }

    // Store the callback globally (increase ref count)
    Py_XINCREF(callback_temp);
    Py_XDECREF(global_callback);
    global_callback = callback_temp;

    // Perform the infinite loop, calling the Python callback each time
    while (1) {
        // Call the callback with no arguments
        PyObject* result = PyObject_CallObject(global_callback, NULL);
        
        // If call returned NULL, an exception occurred
        if (!result) {
            PyErr_Print();  // Print error info to stderr
            break;          // Exit the loop if callback has an error
        }
        Py_DECREF(result);
    }

    // This point is theoretically unreachable unless the callback triggers an error
    Py_RETURN_NONE;
}

// List of methods exposed by this module
static PyMethodDef MyMethods[] = {
    {
        "start_infinite_loop", 
        (PyCFunction)start_infinite_loop, 
        METH_VARARGS,
        "Call the provided callback function endlessly."
    },
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

// Module definition
static struct PyModuleDef myextmodule = {
    PyModuleDef_HEAD_INIT,
    "myext",     // Module name
    NULL,        // Module documentation (may be NULL)
    -1,          // Size of per-interpreter state of the module
    MyMethods
};

// Module initialization function
PyMODINIT_FUNC PyInit_myext(void)
{
    return PyModule_Create(&myextmodule);
}
