using LinearAlgebra
using PyCall
__optimize__ = PyCall.PyNULL()
copy!(__optimize__, pyimport("scipy.optimize"))

function f(x::Vector)
    return norm(x)^2
end

x0 = [0.5, 0]
__optimize__.minimize(f, x0, method="SLSQP")

