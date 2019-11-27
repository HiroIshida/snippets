using JuMP
using OSQP

@time for i in 1:100
    #model = Model(with_optimizer(Ipopt.Optimizer, print_level = 0))
    model = Model(with_optimizer(OSQP.Optimizer))
    set_silent(model)
    @variable(model, x[1:2])
    @objective(model, Min, (x[1] - 1.0)^2 + x[2]^2)
    @constraints model begin
        x[1] >= 2 * x[2]
        x[2] >= 1
    end
    JuMP.optimize!(model)
    println(JuMP.value.(x))
    JuMP.objective_value(model)
end

