import time
import trimesh
import numpy as np
import skrobot
from skrobot.model.primitives import MeshLink

V = np.array([[1., 0, 0], [0., 1., 0.], [0., 0., 1.]])
F = np.array([[0, 1, 2]])
mesh = trimesh.Trimesh(vertices=V, faces=F)
link = MeshLink(visual_mesh=mesh)

viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
viewer.add(link)
viewer.show()

print('==> Press [q] to close window')
while not viewer.has_exit:
    time.sleep(0.1)
    viewer.redraw()
