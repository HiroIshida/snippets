using LinearAlgebra

is_linear = true
A = [0.3 0.7; 0.4 0.8]
x_fixed = [2.0, 2.0]
b = -A * x_fixed + x_fixed
x = [1.0, 1.0]
x_seq = [x]
T = 3
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

#(T * xy_sum - y_sum * x_sum') * inv(T * xx_sum - x_sum * x_sum')
inv(xx_sum) * xy_sum 


