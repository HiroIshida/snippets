import MeshCat
mg_obj = MeshCat.MeshFileObject("./pr2_description/meshes/base_v0/base.obj")
mg_json = MeshCat.lower(mg_obj)

vis = Visualizer()
open(vis)
setobject!(vis, mg_obj)
