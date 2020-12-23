PyCall

```julia
pyfunc(func::Symbol) = dlsym(libpython::Ptr{Void}, func)
```

```julia
pyimport(name::String) =
    PyObject(@pycheckn ccall(pyfunc(:PyImport_ImportModule), PyPtr,
                             (Ptr{Uint8},), bytestring(name)))
```
