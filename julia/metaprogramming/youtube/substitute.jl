function substitute!(ex)
    for (index, arg) in enumerate(ex.args)
        if arg == :x
            ex.args[index] = :y
        end
        if typeof(arg)==Expr
            substitute!(ex.args[index])
        end
    end
end

ex = :(x + x * x)
substitute!(ex)
