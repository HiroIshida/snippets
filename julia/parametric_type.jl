abstract type A end
struct A1 <: A end
struct A2 <: A end

struct B{T<:A}
    type::T
end

sample1(vec::Vector{B{T}}) where T <: A = ()
sample2(vec::Vector{B}) = ()
sample3(vec::Vector{<:B}) = ()

b1 = B(A1())
b2 = B(A2())
vec1 = [b1, b1]
vec2 = [b1, b2]

sample1(vec1)
sample2(vec2)
sample3(vec1)
sample3(vec2)
