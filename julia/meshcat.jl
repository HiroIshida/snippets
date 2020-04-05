using MeshCat
vis = Visualizer()
open(vis)

using GeometryTypes
using CoordinateTransformations

setobject!(vis, HyperRectangle(Vec(0., 0, 0), Vec(1., 1, 1)))
rot = Quat(1., 0, 0, 0)
rot = RotZ(0.4)
trans = Translation(1., 0, 0)
composed = LinearMap(rot) ∘ trans

settransform!(vis, composed)
