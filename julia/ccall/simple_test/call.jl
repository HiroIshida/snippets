cfunc(x) = ccall((:test, "libmylib.so"), Float64, (Float64,), x)
