using LinearAlgebra
using PyCall
__optimize__ = PyCall.PyNULL()
copy!(__optimize__, pyimport("scipy.optimize"))

function f(x::Vector)
    return norm(x)^2
end

function df(x::Vector)
    return 2 * x
end

function h(x::Vector)
    return x[1] - 1
end

function dh(x::Vector)
    return [1, 0]
end

eq_dict = Dict("type"=> "eq", "fun"=> h, "jac"=> dh)
constraints = [eq_dict]
x0 = [0.5, 0]
__optimize__.minimize(f, x0, method="SLSQP", jac=df, constraints=constraints)

