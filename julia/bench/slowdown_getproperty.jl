using BenchmarkTools
using StaticArrays
import Base.*
struct Data
    mat::SMatrix{4, 4, Float64, 16}
end
Data() = Data(ones(SMatrix{4, 4, Float64, 16}))

struct DataWrap
    data::Data
end
DataWrap() = DataWrap(Data())
(*)(data1::Data, data2::Data) = Data(data1.mat*data2.mat)
(*)(data1::Data, d::Float64) = Data(data1.mat*d)

function procedure_slow(dw::DataWrap, data::Data, a)
    return dw.data * data * a
end

function procedure_fast(dw::DataWrap, a)
    return dw.data * dw.data * a
end

function bench1()
    dw = DataWrap()
    for i in 1:10000
        dw = DataWrap(procedure_fast(dw, 0.1))
    end
    return dw
end

function bench2()
    dw = DataWrap()
    for i in 1:10000
        dw = DataWrap(procedure_slow(dw, dw.data, 0.1))
    end
    return dw
end

@benchmark bench1()
@benchmark bench2()




