import trimesh
import pymeshlab
from sdf import sphere, rounded_box, cylinder, rounded_cylinder, box, translate, rotate, X

r = 0.06
f = sphere(r)
f -= sphere(0.05)
f -= box([0.2, 0.2, 0.1]).translate((0, 0, 0.04)).rotate(-0.4, X)

length = 0.3
#handle = rotate(box([0.01, 0.01, length]), 0.3, X)
f |= box([0.02, 0.01, length]).translate((0, -r + 0.005, 0.5 * length))

out = "hoge.stl"
f.save(out, samples=200000)

ms = pymeshlab.MeshSet()
ms.load_new_mesh(out)
ms.meshing_decimation_quadric_edge_collapse(targetfacenum = 1000)
ms.save_current_mesh(out)

mesh = trimesh.load_mesh(out)
scene = mesh.scene()
scene.show()
