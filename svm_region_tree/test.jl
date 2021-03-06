using Revise
using LIBSVM, Test
using DelimitedFiles
using SparseArrays
using Plots
using StaticArrays
using RegionTrees

Revise.includet("adaptive_distance_fields.jl")
#include("adaptive_distance_fields.jl")
using .AdaptivelySampledDistanceFields: ASDF, evaluate

mutable struct Grid2d
  N
  b_min
  b_max
  data
  function Grid2d(;N, b_min, b_max)
    data = nothing
    new(N, b_min, b_max, data)
  end
end

function map(self::Grid2d, f)
  x_lin, y_lin = [range(self.b_min[i], length=self.N+1, stop=self.b_max[i]) for i in [1, 2]]
  plt = plot(xlim=(self.b_min[1], self.b_max[1]), ylim=(self.b_min[2], self.b_max[2]))
  contourf!(plt, x_lin, y_lin, (x, y)->f(x, y))
  return plt
end


function gen_dataset(;N = 100)
  X = randn(2, N)
  Y = [(X[1, i] + X[2, i] > 0) for i in 1:N]
  return X, Y
end

N = 50
X, Y = gen_dataset(N = N)
model = svmtrain(X, Y; verbose=false, probability=true)
g = Grid2d(N = 20, b_min = [-1, -1], b_max = [1, 1])

function f(x)
  X = reshape(x, 2, 1)
  res = svmpredict(model, X)
  return res[2][1]
end

   
function tupler(adf)
  tuple_list = Tuple{Float64, Float64}[]
  for leaf in allleaves(adf)
    data = leaf.data[2:3]
    push!(tuple_list, data)
  end
end

function test()
  tup = nothing
  adf = nothing
  for i in 1:1000
    adf = ASDF(f, SVector(-1.0, -1.0), SVector(2., 2), 0.01, 0.01)
    tup = tupler(adf)
  end
  return adf
end

@time adf = test()

plt = plot(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), legend=nothing)

x = range(-1, stop=1, length=50)
y = range(-1, stop=1, length=50)
contour!(plt, x, y, (x, y) -> evaluate(adf, SVector(x, y)), fill=true)

for leaf in allleaves(adf)
    v = hcat(collect(vertices(leaf.boundary))...)
    plot!(plt, v[1,[1,2,4,3,1]], v[2,[1,2,4,3,1]], color=:white)
end






