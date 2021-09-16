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

using Random

# +
P = 1.0

Random.seed!(123)

n = 250
data = -5.0 .+ collect(1:n) + rand(Normal(0.0, sqrt(P)), n);
# -

function inference(data, x0, c, P)
    n = length(data)
    
    model, (x, y) = smoothing(n, x0, c, P);

    ms_buffer = Vector{Marginal}(undef, n)
    fe_buffer = nothing
    
    fe_subscription = subscribe!(score(BetheFreeEnergy(), model), (fe) -> fe_buffer = fe)
    ms_subscription = subscribe!(getmarginals(x), (ms) -> copyto!(ms_buffer, ms))
    
    update!(y, data)
    
    unsubscribe!(ms_subscription)
    unsubscribe!(fe_subscription)
    
    return ms_buffer, fe_buffer
end

# c[1] is C
# c[2] is μ0
function f(c)
    x0_prior = NormalMeanVariance(c[2], 100.0)
    ms, fe = inference(data, x0_prior, c[1], P)
    return fe
end

using Optim

res = optimize(f, ones(2), GradientDescent(), Optim.Options(g_tol = 1e-3, iterations = 100, store_trace = true, show_trace = true))

res.minimizer # Real values are indeed (c = 1.0 and μ0 = -5.0)


