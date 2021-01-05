using BenchmarkTools

const a = @btime falses(100000);
@btime fill!(a, false)

a = falses(100000)

a = falses(100000)
