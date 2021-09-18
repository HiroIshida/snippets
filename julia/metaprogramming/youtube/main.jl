macro simple(ex)
    return ex
end
@simple Meta.parse("x + y")

macro add_to_multiply(ex)
    return _add_to_multiply(ex)
end

function _add_to_multiply(ex)
    if ex.head == :call && ex.args[1] == :+
        ex.args[1] = :*
    end
    return ex
end

# Quote node and escaping
macro myshow(ex)
    # x = 1
    # @myshow x
    ex2 = QuoteNode(ex)
    :(println($ex2, " = ", $ex))
end

struct Variable
    name::Symbol
end

macro var(ex)
    _var(ex)
end
function _var(ex)
    escaped_ex = esc(ex)
    quoted_ex = QuoteNode(ex)
    
    return :(
        $(escaped_ex) = Variable( $(quoted_ex) )
    )
end

macro var(exs...)
    all_code = quote end
    all_code.args = reduce(vcat, _var(ex) for ex in exs)
    return all_code
end
@var x y z


# multiple arguments
