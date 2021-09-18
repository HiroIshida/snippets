# https://symbolics.juliasymbolics.org/dev/tutorials/symbolic_functions/
using Symbolics
using Distributions

macro tmp(ex)
    _tmp(ex)
end

function _tmp(ex)
    ex_new = quote
        for i in 1:3
            eval($ex)
            println(i)
        end
    end
end

struct Relation
    left::Vector{Symbol}
    right::Vector{Symbol}
    lambda 
end

@variables x y z
dist1 = pdf(Normal(x, 0.1), y)
f_expr = build_function(dist1, [x, y])
dist1_f = eval(f_expr)
dist1_f([0.0, 0.0])
