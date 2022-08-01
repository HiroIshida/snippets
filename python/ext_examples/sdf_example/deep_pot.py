import trimesh
import pymeshlab
from sdf import sphere, rounded_box, cylinder, rounded_cylinder, box, translate

h_cylinder = 0.4
r_cylinder = 0.2
eps_cylinder = 0.03
f = rounded_cylinder(r_cylinder, 0.03, h_cylinder).translate((0, 0, 0.5 * h_cylinder))

# cut the top 
f -= box([0.5, 0.5, 0.1]).translate((0, 0, h_cylinder))

# add the handle
f |= box([0.1, r_cylinder * 2 + 0.1, 0.02]).translate((0, 0, h_cylinder - 0.08))


f_subtract = rounded_cylinder(r_cylinder - eps_cylinder, 0.0, h_cylinder).translate((0, 0, 0.5 * h_cylinder + 2 * eps_cylinder))
f -= f_subtract

out = "hoge.stl"
f.save(out, samples=80000)

ms = pymeshlab.MeshSet()
ms.load_new_mesh(out)
ms.meshing_decimation_quadric_edge_collapse(targetfacenum = 1000)
ms.save_current_mesh(out)

mesh = trimesh.load_mesh(out)
scene = mesh.scene()
scene.show()
