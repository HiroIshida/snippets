using StaticArrays
function f(vec::StaticVector{3, T}) where T<:Real
    return nothing
end

a = MVector{3}(1, 2, 3)
b = SVector{3}(1, 2, 3)
f(a)
f(b)
