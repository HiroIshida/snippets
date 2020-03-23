using Plots
Plots.plotly()

N = 1000
T = zeros(4, N)  # ⊂ R2 ⊗ R2
w_slice = 0.8
for i in 1:N
  b = 0
  d = 0
  while true
    b = rand()*2 - 1
    d = rand()*2 - 1
    abs(b*d- w_slice) < 0.05 && break
  end
  a = rand()*2 - 1
  c = rand()*2 - 1
  t = [a*c, a*d, b*c, b*d]
  T[:, i] = t
end

#plt = scatter(T[1, :], T[2, :], T[3, :], markersize=2)

function constrct_matrix2d(v)
  a, b, c, d = v
  mat = [c 0 a 0; d 0 0 a; 0 c b 0; 0 d 0 b]
  println(rank(mat))
  return mat
end


function constrct_matrix3d(v)
  a, b, c, d, e, f = v
  mat = [
         d 0 0 a 0 0;
         e 0 0 0 a 0;
         f 0 0 0 0 a;
         0 d 0 a 0 0;
         0 e 0 0 a 0;
         0 f 0 0 0 a;
         0 0 d a 0 0;
         0 0 e 0 a 0;
         0 0 f 0 0 a]
  println(rank(mat))
  return mat
end

constrct_matrix3d(randn(6))

 

         

using LinearAlgebra
constrct_matrix([1, 4, 9, 2])

