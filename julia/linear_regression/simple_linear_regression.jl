using LinearAlgebra

is_linear = false

A = [0.3 0.7; 0.4 0.8]
x_fixed = [2.0, 2.0]
b = -A * x_fixed + x_fixed
x = [1.0, 1.0]
x_seq = [x]
T = 30
for i in 1:T-1
    global x
    if is_linear
        x = A * x
    else
        x = A * x + b
    end
    push!(x_seq, x)
end

x_sum = sum(x for x in x_seq[1:end-1])
y_sum = sum(x for x in x_seq[2:end])
xx_sum = sum(x * x' for x in x_seq[1:end-1])
xy_sum = sum(x_seq[t] * x_seq[t+1]' for t in 1:T-1)

A_est = nothing
if is_linear
    A_est = inv(xx_sum) * xy_sum
else
    n = (T-1.0)
    A_est = inv(n * xx_sum - x_sum * x_sum') * (n * xy_sum - x_sum * y_sum')
    B_est = (y_sum - A_est' * x_sum) * (1.0/n)
    println(B_est)
    println(b)
end
@assert A_est' ≈ A
