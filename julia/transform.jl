using LinearAlgebra
import Base.*

Rx(a) = [1 0 0; 0 cos(a) -sin(a); 0 sin(a) cos(a)]
Ry(a) = [cos(a) 0 sin(a); 0 1 0; -sin(a) 0 cos(a)]
Rz(a) = [cos(a) -sin(a) 0; sin(a) cos(a) 0; 0 0 1]

abstract type AbstractRotation end

struct Euler <: AbstractRotation
  rpy
end

function (*)(eul::Euler, p)
  a, b, c = eul.rpy
  mat = Rz(-a) * Ry(-b) * Rx(-c)
  return mat*p
end


struct Transform
  trans
  rot::AbstractRotation
end

function (tf::Transform)(x)
  return tf.trans + tf.rot*x
end
tf = Transform([0, 0, 0], Euler([0, 0, 0.5]))
v = tf([1, 0, 0])
