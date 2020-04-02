struct Rectangle
  tf
  b
end

function compute_sd(obj::Rectangle, p)
  q = abs.(p) - obj.b
  left = norm(map(x->(max(x, 0.)), q))
  right = min(max(q[1], max(q[2], q[3])), 0.)
  return left + right
end
