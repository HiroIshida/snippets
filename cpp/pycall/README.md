PyCall

```julia
pyfunc(func::Symbol) = dlsym(libpython::Ptr{Void}, func)
```

```julia
pyimport(name::String) =
    PyObject(@pycheckn ccall(pyfunc(:PyImport_ImportModule), PyPtr,
                             (Ptr{Uint8},), bytestring(name)))
```

Notes :
In python3x, `PySys_SetPath` takes `wchat_t*` type as a parameter. But is python2x, it takes normal `char*`. As for converting `char*` to `wchar_t*` please see : https://stackoverflow.com/questions/8032080/how-to-convert-char-to-wchar-t
