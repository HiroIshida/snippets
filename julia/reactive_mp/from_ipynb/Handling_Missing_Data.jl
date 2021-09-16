# -*- coding: utf-8 -*-
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
@rule NormalMeanVariance(:μ, Marginalisation) (m_out::Any, m_v::Missing) = missing
@rule NormalMeanVariance(:μ, Marginalisation) (m_out::Missing, m_v::Any) = missing

@rule typeof(+)(:in1, Marginalisation) (m_out::Missing, m_in2::Any) = missing
@rule typeof(+)(:in1, Marginalisation) (m_out::Any, m_in2::Missing) = missing

# +
P = 1.0

n = 500
data = convert(Vector{Union{Float64, Missing}}, collect(1:n) + rand(Normal(0.0, sqrt(P)), n));

for index in map(d -> rem(abs(d), n), rand(Int, Int(n / 2)))
    data[index] = missing
end
# -

function inference(data, x0, P)
    n = length(data)
    
    _, (x, y) = smoothing(n, x0, P);

    buffer    = Vector{Marginal}(undef, n)
    marginals = getmarginals(x)
    
    subscription = subscribe!(marginals, (ms) -> copyto!(buffer, ms))
    
    update!(y, data)
    
    unsubscribe!(subscription)
    
    return buffer
end

x0_prior = NormalMeanVariance(0.0, 1000.0)
res = inference(data, x0_prior, P)


