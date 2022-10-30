import time
import numpy as np
import skrobot
from skrobot.model.primitives import Box, Axis
from skrobot.sdf.signed_distance_function import SignedDistanceFunction
from skrobot.coordinates import Transform

depth = 0.6
width = 0.8
height = 0.8
box = Box(extents=[depth, width, height])
box.translate([0.5, 0.2, 0.5 * height])
box.rotate(0.3, axis="z")

tf = box.get_transform()

#grid_sizes = [40, 60, 20]
mesh_height = 0.3
grid_sizes = [4, 6, 3]
xlin = np.linspace(-0.5 * depth, 0.5 * depth, grid_sizes[0])
ylin = np.linspace(-0.5 * width, 0.5 * width, grid_sizes[1])
zlin = np.linspace(0.5 * height, 0.5 * height + mesh_height, grid_sizes[2])
X, Y, Z = np.meshgrid(xlin, ylin, zlin)
pts_local = np.array(list(zip(X.flatten(), Y.flatten(), Z.flatten())))
pts_world = tf.transform_vector(pts_local)
axis_list = [Axis(axis_radius=0.01, axis_length=0.02, pos=vec) for vec in pts_world]

viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
viewer.add(box)
for axis in axis_list:
    viewer.add(axis)
viewer.show()

print("==> Press [q] to close window")
while not viewer.has_exit:
    time.sleep(0.1)
    viewer.redraw()
