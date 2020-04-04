struct Euler{T} #<: AbstractRotation
  rpy::SVector{3, T}
  mat::SMatrix{3, 3, T}
end

function Euler(r, p, y)
  T = promote_type(typeof(r), typeof(p), typeof(y))
  rpy = SVector{3, T}(r, p, y)
  mat = Rz(-r) * Ry(-p) * Rx(-y)
  Euler{T}(rpy, mat)
end
