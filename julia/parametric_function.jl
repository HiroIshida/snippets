function hoge(::Type{T}) where T<:Real
    println(T)
end
hoge(Float64)
hoge(Int64)
