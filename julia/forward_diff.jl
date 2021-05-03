using StaticArrays
using LinearAlgebra
using ForwardDiff
const SVector3f = SVector{3, Float64}

struct A
    a::Float64
end
(a::A)(x::Vector) = (x[1]^2 + x[2]^2)*a.a
(a::A)(x::SVector{3, <:Any}) = (x[1]^2 + x[2]^2 + x[3]^2) * a.a

f = A(2.0)
x = [2., 1.]
grad = ForwardDiff.gradient(f, x)

sx = SVector3f(3., 1., 0.0)
grad = ForwardDiff.gradient(f, sx)

