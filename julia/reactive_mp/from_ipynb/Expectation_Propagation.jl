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

# # Expectation Propagation

# ReactiveMP comes with support for expectation propagation (EP). In this demo we illustrate EP in the context of state-estimation in a linear state-space model that combines a Gaussian state-evolution model with a discrete observation model. Here, the probit function links continuous variable $x_t$ with the discrete variable $y_t$. The model is defined as:

# \begin{align}
#     u &= 0.1 \\
#     x_0 &\sim \mathcal{N}(0, 100) \\
#     x_t &\sim \mathcal{N}(x_{t-1}+ u, 0.01) \\
#     y_t &\sim \mathrm{Ber}(\Phi(x_t))
# \end{align}

# ## Import packages

# +
using Pkg
Pkg.activate("..")

using StatsFuns: normcdf
using Random
using PyPlot
using Revise

using ReactiveMP
using GraphPPL
using Rocket
# -

# ##  Data generation

# +
function generate_data(nr_samples::Int64)
    
    # hyper parameters
    u = 0.1

    # allocate space for data
    data_x = zeros(nr_samples + 1)
    data_y = zeros(nr_samples)
    
    # initialize data
    data_x[1] = -2
    
    # generate data
    for k = 2:nr_samples + 1
        
        # calculate new x
        data_x[k] = data_x[k-1] + u + sqrt(0.01)*randn()
        
        # calculate y
        data_y[k-1] = normcdf(data_x[k]) > rand()
        
    end
    
    # return data
    return data_x, data_y
    
end;
# -

n = 40

data_x, data_y = generate_data(n);

plt.figure()
plt.scatter(1:n, data_y, label="y")
plt.plot(data_x, label="x")
plt.xlabel("t")
plt.ylabel("x, y")
plt.legend()
plt.grid();

# ## Model specification

# +
@model function probit_model(nr_samples::Int64)
    
    # allocate space for variables
    x = randomvar(nr_samples + 1)
    y = datavar(Float64, nr_samples)
    
    # specify uninformative prior
    x[1] ~ NormalMeanPrecision(0.0, 0.01)
    
    # create model 
    for k = 2:nr_samples + 1
        x[k] ~ NormalMeanPrecision(x[k - 1] + 0.1, 100)
        y[k - 1] ~ Probit(x[k]) where {
            # Probit node by default uses RequireInbound pipeline with vague(NormalMeanPrecision) message as initial value for `in` edge
            # To change initial value use may specify it manually, like
            # pipeline = RequireInbound(in = NormalMeanPrecision(0, 1.0)) 
        }
    end
    
    # return parameters
    return x, y
    
end;
# -

# ## Inference

# +
function inference(data_y; nr_iterations::Int64=10)
    
    # fetch number of samples
    nr_samples = length(data_y)
    
    # create model 
    model, (x, y) = probit_model(nr_samples)
    
    # allocate marginals and free energy
    mx = buffer(Marginal, nr_samples + 1)
    fe = []
    
    # subscribe to marginals
    x_sub = subscribe!(getmarginals(x), mx)

    # update data (i.e. run inference)
    for it = 1:nr_iterations
        update!(y, data_y)
    end
    
    # unsubscribe
    unsubscribe!(x_sub)

    # return marginals
    return mx
    
end;
# -

mx = inference(data_y);

# ## Results

plt.figure()
plt.scatter(1:n, data_y, label="y", color="blue")
plt.plot(data_x, label="x (true)", color="blue")
plt.plot(mean.(mx), label="x (inferred mean)", color="orange")
plt.fill_between(0:n, mean.(mx) .- std.(mx), mean.(mx) + std.(mx), label="x (inferred ±σ)", alpha=0.3, color="orange")
plt.xlabel("t")
plt.ylabel("x, y")
plt.legend()
plt.grid();




