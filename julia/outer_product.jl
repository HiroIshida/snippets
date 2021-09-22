using Test
using LinearAlgebra
# https://stackoverflow.com/questions/38773076/what-is-the-fastest-way-to-compute-the-sum-of-outer-products-julia
N = 10000
Y = randn(3, N)
weight = randn(N)

function naive(Y, N, weight)
    a = zeros(3, 3)
    for i in 1:N
        @views a += Y[:, i] * Y[:, i]' * weight[i]
    end
    return a
end

function faster(Y, weight)
    return Y * Diagonal(weight) * Y'
end

@test isapprox(naive(Y, N, weight), faster(Y, weight), atol=1e-5)

using BenchmarkTools
@btime naive(Y, N, weight) # 1.117 ms => 1.553 ms (with weight)
@btime faster(Y, weight) # 68.5 μs => 119 μs (with weight)
