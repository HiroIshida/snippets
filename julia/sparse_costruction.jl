using BenchmarkTools
using SparseArrays
using LinearAlgebra

function test()
    A = zeros(1000, 1000)
    function f1()
        SparseMatrixCSC(A)
    end

    B = Diagonal(zeros(1000))
    function f2()
        SparseMatrixCSC(B)
    end
    @btime f1()
    @btime f2()
end

