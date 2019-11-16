mutable struct Bin
    x_min
    x_max
    children
    idx
end

function Bin(x_min, x_max)
    Bin(x_min, x_max, nothing, nothing)
end

function split!(bin::Bin)
    x_mid = (bin.x_min + bin.x_max) * 0.5
    child_lower = Bin(bin.x_min, x_mid)
    child_upper = Bin(x_mid, bin.x_max)
    bin.children = (child_lower, child_upper)
    return bin.children
end

root = Bin(0, 1.0)
function f!(bin::Bin, depth_max = 10)
    idx = 0
    function recursion(bin::Bin; depth = 0)
        bin.idx = idx
        idx += 1
        (depth == depth_max) && return
        for child in split!(bin)
            recursion(child; depth = depth + 1)
        end
    end
    recursion(bin)
end

f!(root)

