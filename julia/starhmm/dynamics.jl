using PyPlot
using LinearAlgebra

mutable struct State 
    x::Vector
    v::Vector
end

struct Attractor
    center::Vector
    k::Float64
    c::Float64
end
function propagate(attr::Attractor, state::State, dt)
    force = - attr.k * (state.x - attr.center) - attr.c * state.v
    state.v += force * dt 
    state.x += state.v * dt
    return state
end

abstract type GoalRegion end
isInside(gr::GoalRegion, state::State) = error("implement this")

struct SimpleGoalRegion <: GoalRegion
    attr::Attractor
    r::Float64
end
isInside(gr::SimpleGoalRegion, state::State) = (LinearAlgebra.norm(gr.attr.center - state.x) < gr.r)

struct SequentialAttractor
    attrs::Vector{Attractor}
    grs::Vector{GoalRegion}
end
function propagate(seqattr::SequentialAttractor, state::State, phase::Int, dt)
    attr = seqattr.attrs[phase]
    state = propagate(attr, state, dt) 

    phase_new = phase
    if length(seqattr.grs) >= phase
        gr = seqattr.grs[phase]
        if isInside(gr, state)
            println("hoge")
            phase_new += 1
        end
    end
    return state, phase_new
end

function main()
    attr1 = Attractor([0, 1.], 0.2, 0.2)
    attr2 = Attractor([1.5, 1.], 0.2, 0.2)
    gr1 = SimpleGoalRegion(attr1, 0.05)
    seq_attrs = SequentialAttractor([attr1, attr2], [gr1])

    states = []
    s = State([0, 0.], [0.1, 0.])
    phase = 1
    for i in 1:500
        push!(states, deepcopy(s))
        s, phase = propagate(seq_attrs, s, phase, 0.1)
    end
    x_seq = [s.x[1] for s in states]
    y_seq = [s.x[2] for s in states]
    plot(x_seq, y_seq)
end
