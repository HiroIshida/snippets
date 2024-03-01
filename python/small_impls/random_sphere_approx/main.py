import logging

import trimesh

from mesh2spheres.algorithm import find_sphere_approximation
from mesh2spheres.utils import GridSDF
import ycb_utils

logging.basicConfig(level=logging.INFO)

mesh = ycb_utils.load("019_pitcher_base")
sdf = GridSDF(mesh, n_grid=100, n_padding=20)

sphere_params = find_sphere_approximation(mesh, sdf, 0.001, coverage_tolerance=0.005)
sphere_list = [mesh]
for p, r in sphere_params:
    sphere = trimesh.creation.uv_sphere(radius=r)
    sphere.visual.face_colors = [255, 0, 0, 200]
    sphere.apply_translation(p)
    sphere_list.append(sphere)
scene = trimesh.Scene(sphere_list)
scene.show()
