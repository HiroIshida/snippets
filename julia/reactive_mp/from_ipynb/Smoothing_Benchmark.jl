# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Julia 1.6.2
#     language: julia
#     name: julia-1.6
# ---

using Rocket
using ReactiveMP
using GraphPPL
using BenchmarkTools
using Distributions
using Random

@model function smoothing(n, x0, c::ConstVariable, P::ConstVariable)
    
    x_prior ~ NormalMeanVariance(mean(x0), cov(x0)) 

    x = randomvar(n)
    y = datavar(Float64, n)
    
    x_prev = x_prior
    
    for i in 1:n
        x[i] ~ x_prev + c
        y[i] ~ NormalMeanVariance(x[i], P)
        
        x_prev = x[i]
    end

    return x, y
end

# +
seed = 123

rng = MersenneTwister(seed)

P = 1.0
n = 500

data = collect(1:n) + rand(rng, Normal(0.0, sqrt(P)), n);
# -

function inference(data, x0, P)
    n = length(data)
    
    _, (x, y) = smoothing(n, x0, 1.0, P);

    x_buffer  = buffer(Marginal, n)
    marginals = getmarginals(x)
    
    subscription = subscribe!(marginals, x_buffer)
    
    update!(y, data)
    
    unsubscribe!(subscription)
    
    return getvalues(x_buffer)
end

x0_prior = NormalMeanVariance(0.0, 10000.0)

@benchmark res = inference($data, $x0_prior, $P)

inference(data, x0_prior, P)


