const mylib = joinpath(pwd(), "libmylib.so")
cfunc(x) = ccall((:test, mylib), Float64, (Float64,), x)
cfunc(2.0)
cfunc2(f) = ccall((:test_func_pointer, mylib), Cvoid, (Ptr{Cvoid},), f)

function closure()
    b = 0 
    function func(x)
        b = x
        nothing
    end
    function show()
        println(b)
    end
    return func, show
end

func, show = closure()
func_c = @cfunction(func, Cvoid, (Cdouble,))
cfunc2(func_c)
