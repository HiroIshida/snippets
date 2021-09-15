using ForneyLab

function generate_sample(n_sample)
    A_data = [0.9 0.0 0.1; 0.1 0.9 0.0; 0.0 0.1 0.9] # Transition probabilities (some transitions are impossible)
    B_data = [0.9 0.05 0.05; 0.05 0.9 0.05; 0.05 0.05 0.9] # Observation noise

    s_0_data = [1.0, 0.0, 0.0] # Initial state

    # Generate some data
    s_data = Vector{Vector{Float64}}(undef, n_samples) # one-hot encoding of the states
    x_data = Vector{Vector{Float64}}(undef, n_samples) # one-hot encoding of the observations
    s_t_min_data = s_0_data
    for t = 1:n_samples a = A_data*s_t_min_data
        s_data[t] = sample(ProbabilityDistribution(Categorical, p=a./sum(a))) # Simulate state transition
        b = B_data*s_data[t]
        x_data[t] = sample(ProbabilityDistribution(Categorical, p=b./sum(b))) # Simulate observation
        
        s_t_min_data = s_data[t]
    end
    s_data, x_data
end


function hmmcodegen(batch_size)
    g = FactorGraph()

    # clamp placeholders to pass priors as data
    @RV A ~ Dirichlet(placeholder(:A_0, dims=(3, 3))) 
    @RV B ~ Dirichlet(placeholder(:B_0, dims=(3, 3)))
    @RV s_0 ~ Categorical(1/3*ones(3))

    s = Vector{Variable}(undef, batch_size) # one-hot coding
    x = Vector{Variable}(undef, batch_size) # one-hot coding
    s_t_min = s_0
    for t in 1:batch_size
        @RV s[t] ~ Transition(s_t_min, A)
        @RV x[t] ~ Transition(s[t], B)
        
        s_t_min = s[t]
        
        placeholder(x[t], :x, index=t, dims=(3,))
    end;

    pfz = PosteriorFactorization(A, B, [s_0; s], ids=[:A, :B, :S])
    algo = messagePassingAlgorithm(free_energy=true)
    source_code = algorithmSourceCode(algo, free_energy=true);
end

function learn_hmm(A_0_prev, B_0_prev, x_data, batch_size, n_its=10)
    n_samples = length(x_data)

    n_its = 10

    s_ = []
    A_ = []
    B_ = []
    n_batch = Int(n_samples/batch_size)
    F = Matrix{Float64}(undef, n_batch, n_its)

    for i in 1:batch_size:n_samples-batch_size
        data = Dict(:x => x_data[i:i+batch_size], :A_0 => A_0_prev, :B_0 => B_0_prev)
        marginals = Dict{Symbol, ProbabilityDistribution}(:A => vague(Dirichlet, (3,3)), :B => vague(Dirichlet, (3,3)))
        for v in 1:n_its
            stepS!(data, marginals)
            stepB!(data, marginals)
            stepA!(data, marginals)

            # Compute FE for every batch
            F[div(i, batch_size)+1, v] = freeEnergy(data, marginals)
        end

        # Extract posteriors
        A_0_prev = marginals[:A].params[:a]
        B_0_prev = marginals[:B].params[:a]

        # Save posterior marginals for each batch
        push!(s_, [marginals[:s_*t] for t in 1:batch_size])
        push!(A_, marginals[:A])
        push!(B_, marginals[:B])
    end
    return s_, A_, B_
end

n_samples = 600

s_data, x_data = generate_sample(n_samples)
println("data generated")

batch_size = 100
source_code = hmmcodegen(batch_size)
eval(Meta.parse(source_code))
println("source code generated")

# Define values for prior statistics
A_0_prev = ones(3, 3)
B_0_prev = [10.0 1.0 1.0; 1.0 10.0 1.0; 1.0 1.0 10.0]
Sseqs, A, B = learn_hmm(A_0_prev, B_0_prev, x_data, batch_size)

y_true = [Float64(argmax(s)) - 1.0 for s in s_data]
tmp = []
for Sseq in Sseqs
    tmp = vcat(tmp, [Float64(argmax(s.params[:p])) - 1.0 for s in Sseq])
end
using Plots
plot(y_true)
plot!(tmp)

