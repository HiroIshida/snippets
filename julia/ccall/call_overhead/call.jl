cfunc(x) = ccall((:test, "libmylib.so"), Cvoid, (Float64,), x)
