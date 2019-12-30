
function ordered_grid_gen(n_depth)
  x_lst = []
  for i in 1:n_depth
    append!(x_lst, progressive_grid_gen(i))
  end
  return x_lst
end

function progressive_grid_gen(n_depth)

  N = 2^(n_depth-1)+1
  x_lst_pre = []
  for i in 0:(N-1)
    for j in 0:(N-1)
      push!(x_lst_pre, [i, j])
    end
  end

  n_depth == 1 && return(x_lst_pre)

  x_lst = []
  for x in x_lst_pre # filtering
    if sum(x.%2) > 0
      push!(x_lst, x./(N-1))
    end
  end
  return x_lst
end

using Plots
function convert(x_lst)
  n = length(x_lst)
  X1 = [x_lst[i][1] for i in 1:n]
  X2 = [x_lst[i][2] for i in 1:n]
  return hcat(X1, X2)
end

a1 = progressive_grid_gen(1)
a2 = progressive_grid_gen(2)

x_lst_lst = [convert(progressive_grid_gen(i)) for i in 1:4]
scatter(x_lst_lst[1][:, 1], x_lst_lst[1][:, 2], color = "red")
scatter!(x_lst_lst[2][:, 1], x_lst_lst[2][:, 2], color = "blue")
scatter!(x_lst_lst[3][:, 1], x_lst_lst[3][:, 2], color = "green")
scatter!(x_lst_lst[4][:, 1], x_lst_lst[4][:, 2], color = "yellow")

ordered_grid_gen(4)
