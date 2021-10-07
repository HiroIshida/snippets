PyCall

```julia
pyfunc(func::Symbol) = dlsym(libpython::Ptr{Void}, func)
```

```julia
pyimport(name::String) =
    PyObject(@pycheckn ccall(pyfunc(:PyImport_ImportModule), PyPtr,
                             (Ptr{Uint8},), bytestring(name)))
```

Imported module is stored in `PyObject`. Then, module function can be called by
```julia
(o::PyObject)(args...; kwargs...) =
    return convert(PyAny, _pycall!(PyNULL(), o, args, kwargs))
```

```julia
function __pycall!(ret::PyObject, pyargsptr::PyPtr, o::Union{PyObject,PyPtr},
  kw::Union{Ptr{Cvoid}, PyObject})
    disable_sigint() do
        retptr = @pycheckn ccall((@pysym :PyObject_Call), PyPtr, (PyPtr,PyPtr,PyPtr), o,
                        pyargsptr, kw)
        pydecref_(ret)
        setfield!(ret, :o, retptr)
    end
    return ret #::PyObject
end
```

Notes :
In python3x, `PySys_SetPath` takes `wchat_t*` type as a parameter. But is python2x, it takes normal `char*`. As for converting `char*` to `wchar_t*` please see : https://stackoverflow.com/questions/8032080/how-to-convert-char-to-wchar-t
