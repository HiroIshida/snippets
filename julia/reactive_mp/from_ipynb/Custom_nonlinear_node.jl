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
#     display_name: Julia Precompiled 1.6.2
#     language: julia
#     name: julia-precompiled-1.6
# ---

using ReactiveMP
using Distributions
using Random
using BenchmarkTools
using Rocket
using GraphPPL

# Here is an example of creating custom node with nonlinear function approximation with samplelist.

# ### Custom node creation

# +
struct NonlinearNode end # Dummy structure just to make Julia happy

struct NonlinearMeta{F}
    fn       :: F   # Nonlinear function, we assume 1 float input - 1 float ouput
    nsamples :: Int # Number of samples used in approximation
end
# -

@node NonlinearNode Deterministic [ out, in ]

# We need to define two Sum-product message computation rules for our new custom node
# - Rule for outbound message on `out` edge given inbound message on `in` edge
# - Rule for outbound message on `in` edge given inbound message on `out` edge
# - Both rules accept optional meta object

# Rule for outbound message on `out` edge given inbound message on `in` edge
@rule NonlinearNode(:out, Marginalisation) (m_in::NormalMeanVariance, meta::NonlinearMeta) = begin 
    samples = rand(m_in, meta.nsamples)
    return SampleList(map(meta.fn, samples))
end

# Rule for outbound message on `in` edge given inbound message on `out` edge
@rule NonlinearNode(:in, Marginalisation) (m_out::Gamma, meta::NonlinearMeta) = begin     
    return ContinuousUnivariateLogPdf((x) -> logpdf(m_out, meta.fn(x)))
end

# ### Model specification
#
# After we have defined our custom node with custom rules we may proceed with a model specification. In this model the aim is to estimate a precision of a Gaussian distribution given IID samples. We pass a random variable `θ` through a non-linear transformation `fn` to make it positive and suitable for a precision parameter of a Gaussian distribution. We later on will estimate posterior of `θ`.

@model [ default_factorisation = MeanField() ] function nonlinear_estimation(fn::Function, n, nsamples)
    
    # We fallback to generic product and sample list approximation 
    # with prior as proposal distribution
    θ = randomvar() where {
        prod_constraint = ProdGeneric(),
        form_constraint = SampleListFormConstraint(nsamples, LeftProposal()) 
    }
    
    θ ~ NormalMeanVariance(0.0, 100.0)
    m ~ NormalMeanVariance(0.0, 1.0)
    
    # We fallback to generic product and sample list approximation 
    # with likelihood as proposal distribution
    w = randomvar() where {
        prod_constraint = ProdGeneric(),
        form_constraint = SampleListFormConstraint(nsamples, RightProposal()) 
    }
    
    w ~ NonlinearNode(θ) where { meta = NonlinearMeta(fn, nsamples) }
    
    y = datavar(Float64, n)
    
    for i in 1:n
        y[i] ~ NormalMeanPrecision(m, w)
    end
    
    return θ, m, w, y
end

function inference(fn::Function, data, nsamples, niters)
    n = length(data)
    
    model, (θ, m, w, y) = nonlinear_estimation(fn, n, nsamples)
    
    θkeep = keep(Marginal)
    subθ  = subscribe!(getmarginal(θ), θkeep)
    
    setmarginal!(m, vague(NormalMeanPrecision))
    setmarginal!(w, vague(Gamma))
    
    for i in 1:niters
        update!(y, data)
    end
    
    return θkeep
end

# Here we generate some data

function nonlinear_fn(x) # We use exp as a simple example
    return exp(x + 2)
end

# +
seed = 123
rng  = MersenneTwister(seed)

niters   = 15 # Number of VMP iterations
nsamples = 5_000 # Number of samples in approximation

n = 1000 # Number of IID samples
μ = -10.0
θ = -3.0
w = nonlinear_fn(θ)


data = rand(rng, NormalMeanPrecision(μ, w), n);
# -

@time θposterior = inference(nonlinear_fn, data, nsamples, niters);

using CairoMakie

let
    f = Figure()
    
    θplot = Axis(f[1, 1])
    
    θplot.title = "Posterior for θ over VMP iterations"
    
    range = 1:niters
    
    lines!(θplot, range, mean.(θposterior), label = "Estimated")
    band!(θplot, range, mean.(θposterior) .- std.(θposterior), mean.(θposterior) .+ std.(θposterior))
    
    lines!(θplot, range, fill(θ, niters), label = "Real")
    
    axislegend(θplot, position = :rb)
    
    ylims!(θplot, (-10, -2))
    
    f
end

@benchmark inference($nonlinear_fn, $data, $nsamples, $niters)

versioninfo()


