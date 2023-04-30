from sklearn import svm
from skimage import measure
import numpy as np
import trimesh

def is_inside(pos):
    x, y, z = pos
    value = x ** 2 + (y / 2) ** 2 + (z / 3) ** 2
    return value < 1.0

# create dataset and fit svm
X = np.random.rand(20000, 3) * 8 - np.ones(3) * 4
bools = []
for pos in X:
    bools.append(is_inside(pos))
Y = np.array(bools).astype(float)

clf = svm.SVC(C=100)
clf.fit(X, Y)

# evaluate grid
N = 30
xlin, ylin, zlin = [np.linspace(-4, 4, N) for _ in range(3)]
X, Y, Z = np.meshgrid(xlin, ylin, zlin)
pts = np.array(list(zip(X.flatten(), Y.flatten(), Z.flatten())))

fs = clf.decision_function(pts)

spacing = np.ones(3) * 8/(N-1)
F = fs.reshape(N, N, N)
F = np.swapaxes(F, 0, 1) # important!!!
verts, faces, _, _ = measure.marching_cubes_lewiner(F, 0, spacing=spacing)
verts = verts - np.ones(3) * 4
mesh = trimesh.Trimesh(vertices=verts, faces=faces)

debug_by_matplotlib = False
if debug_by_matplotlib:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2], alpha=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()
else:
    # scene = mesh.scene()
    # png = scene.save_image(resolution=[400, 200], visible=True)
    # with open("tmp.png", 'wb') as f:
    #     f.write(png)
    #     f.close()
    import time
    from skrobot.model.primitives import MeshLink
    from skrobot.viewers import TrimeshSceneViewer
    mesh_link = MeshLink(mesh)
    mesh_link.visual_mesh.visual.face_colors[:, 0] = 255
    mesh_link.visual_mesh.visual.face_colors[:, 1] = 0
    mesh_link.visual_mesh.visual.face_colors[:, 2] = 0
    mesh_link.visual_mesh.visual.face_colors[:, 3] = 180
    vis = TrimeshSceneViewer()
    vis.add(mesh_link)
    vis.redraw()
    vis.show()
    time.sleep(100)
