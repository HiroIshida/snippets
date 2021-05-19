using BenchmarkTools
using ResumableFunctions
# compare speed of channel and pure list 
function channel_bench()
    Channel() do c
        for i in 1:20
            put!(c, (i, i+1))
        end
    end
end

function list_bench()
    return [(i, i+1) for i in 1:20]
end

@resumable function generator_bench()
    for i in 1:20
        @yield (i, i+1)
    end
end

@btime channel_bench() # 793.623 ns
@btime list_bench() # 43.37 ns
@btime generator_bench() # 4.624 ns
