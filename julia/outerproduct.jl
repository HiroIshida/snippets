using Test
# https://stackoverflow.com/questions/38773076/what-is-the-fastest-way-to-compute-the-sum-of-outer-products-julia
N = 10000
Y = randn(3, N)

function naive(Y, N)
    a = zeros(3, 3)
    for i in 1:N
        @views a += Y[:, i] * Y[:, i]'
    end
    return a
end

function faster(Y)
    return Y * Y'
end

@test isapprox(naive(Y, N), faster(Y), atol=1e-5)

using BenchmarkTools
@btime naive(Y, N) # 1.117 ms
@btime faster(Y) # 68.5 μs
