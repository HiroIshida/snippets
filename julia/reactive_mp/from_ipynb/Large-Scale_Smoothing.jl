# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Julia 1.6.1
#     language: julia
#     name: julia-1.6
# ---

using Rocket
using ReactiveMP
using GraphPPL
using BenchmarkTools
using Distributions
using MacroTools

@model function smoothing(n, x0, P::ConstVariable)
    
    x_prior ~ NormalMeanVariance(mean(x0), cov(x0)) 

    x = randomvar(n)
    y = datavar(Float64, n)
    c = constvar(1.0)

    x_prev = x_prior

    for i in 1:n
        x[i] ~ x_prev + c
        y[i] ~ NormalMeanVariance(x[i], P)
        
        x_prev = x[i]
    end

    return x, y
end

# +
P = 1.0

n = 10_000
k = 500
data = collect(1:n) + rand(Normal(0.0, sqrt(P)), n);
# -

function inference(; data, k, x0, P)
    n = length(data)
    
    _, (x, y) = smoothing(n, x0, P, options = (limit_stack_depth = k, ));

    buffer    = Vector{Marginal}(undef, n)
    marginals = getmarginals(x)
    
    subscription = subscribe!(marginals, (ms) -> copyto!(buffer, ms))
    
    update!(y, data)
    
    unsubscribe!(subscription)
    
    return buffer
end

@benchmark res = inference(
    data = $data,
    k = $k,
    x0 = NormalMeanVariance(0.0, 10000.0),
    P = $P
)

@benchmark res = inference(
    data = $data,
    k = $k,
    x0 = NormalMeanVariance(0.0, 10000.0),
    P = $P
)


