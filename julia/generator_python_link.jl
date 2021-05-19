using BenchmarkTools
using ResumableFunctions
# compare speed of channel and pure list 
function channel_bench()
    Channel() do c
        for i in 1:20
            put!(c, i)
        end
    end
end

function list_bench()
    return [i for i in 1:20]
end

@resumable function generator_bench()
    for i in 1:20
        @yield i
    end
end

@btime sum(channel_bench())
@btime sum(list_bench())
@btime sum(generator_bench())

#=
62.987 μs (43 allocations: 1.69 KiB)
42.855 ns (1 allocation: 240 bytes)
28.343 ns (1 allocation: 48 bytes)
=#
