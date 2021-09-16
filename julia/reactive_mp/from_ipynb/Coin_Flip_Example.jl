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
#     display_name: Julia 1.6.2
#     language: julia
#     name: julia-1.6
# ---

using Rocket, GraphPPL, ReactiveMP, Distributions, Random

# +
rng = MersenneTwister(42)
n = 500
p = 0.75
distribution = Bernoulli(p)

dataset = float.(rand(rng, Bernoulli(p), n));
# -

# GraphPPL.jl export `@model` macro for model specification
# It accepts a regular Julia function and builds an FFG under the hood
@model function coin_model(n)

    # `datavar` creates data 'inputs' in our model
    # We will pass data later on to these inputs
    # In this example we create a sequence of inputs that accepts Float64
    y = datavar(Float64, n)

    # We endow θ parameter of our model with some prior
    θ ~ Beta(2.0, 7.0)

    # We assume that outcome of each coin flip is governed by the Bernoulli distribution
    for i in 1:n
        y[i] ~ Bernoulli(θ)
    end

    # We return references to our data inputs and θ parameter
    # We will use these references later on during inference step
    return y, θ
end

function inference(data)
    n = length(data)

    # `coin_model` function from `@model` macro returns a reference to the model object and
    # the same output as in `return` statement in the original function specification
    model, (y, θ) = coin_model(n)

    # Reference for future posterior marginal
    mθ = nothing

    # `getmarginal` function returns an observable of future posterior marginal updates
    # We use `Rocket.jl` API to subscribe on this observable
    # As soon as posterior marginal update is available we just save it in `mθ`
    subscription = subscribe!(getmarginal(θ), (m) -> mθ = m)

    # `update!` function passes data to our data inputs
    update!(y, data)

    # It is always a good practice to unsubscribe and to
    # free computer resources held by the subscription
    unsubscribe!(subscription)

    # Here we return our resulting posterior marginal
    return mθ
end

θestimated = inference(dataset)

# +
using Plots, LaTeXStrings; theme(:default)

rθ = range(0, 1, length = 1000)

p1 = plot(rθ, (x) -> pdf(Beta(2.0, 7.0), x), title="Prior", fillalpha=0.3, fillrange = 0, label="P(θ)", c=1,)
p2 = plot(rθ, (x) -> pdf(θestimated, x), title="Posterior", fillalpha=0.3, fillrange = 0, label="P(θ|y)", c=3)

p = plot(p1, p2, layout = @layout([ a; b ]))
