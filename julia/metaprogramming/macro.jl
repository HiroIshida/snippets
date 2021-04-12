# NOTE : in practice this kind of manipulation should be done by
# using a higher order function

function constraint(x::Vector)
    val = sum(x .* 2)
    grad = 2 * x
    return val, grad
end

macro nlopt_const(ex)
    :(function $(ex)(x, grad)
          val_, grad_ = $(ex)(x)
          copy!(grad, grad_)
          return val_
      end)
end

x = [1, 2]
grad = zeros(2)
@nlopt_const constraint
nlopt_const(x, grad)
