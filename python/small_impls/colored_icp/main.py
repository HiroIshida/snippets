from scipy.spatial import KDTree
import numpy as np
from skrobot.coordinates import Coordinates
from skrobot.model.link import Link
from skrobot.viewers import TrimeshSceneViewer
from trimesh import PointCloud

class ColoredPointCloud(Link):
    def __init__(self, pts, colors, name=None):
        super().__init__(name=name)
        self.pts = pts
        self.colors = colors
        self._visual_mesh = PointCloud(pts, colors)


src = np.load('point_cloud_colored1_filtered.npy')
src_pts = src[:, :3]
src_colors = src[:, 3:]

target = np.load('point_cloud_colored2_filtered.npy')
# target = src.copy()
target_pts = target[:, :3]
target_colors = target[:, 3:]

tree = KDTree(target_pts)

def get_co(planer_pose: np.ndarray):
    x, y, yaw = planer_pose
    co = Coordinates(pos=[x, y, 0], rot=[yaw, 0, 0.0])
    return co

def loss(planer_pose: np.ndarray):
    co = get_co(planer_pose)
    src_pts_converted = co.transform_vector(src_pts)
    _, indices = tree.query(src_pts_converted)
    target_pts_nearest = target_pts[indices]
    assert src_pts_converted.shape == target_pts_nearest.shape

    loss = np.sum(np.linalg.norm(src_pts_converted - target_pts_nearest, axis=1))
    return loss

# optimize
from scipy.optimize import minimize
res = minimize(loss, np.array([0, 0, 0]), method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
print(res.x)

co = get_co(res.x)
src_pts_converted = co.transform_vector(src_pts)

src_link = ColoredPointCloud(src_pts_converted, src_colors)
target_link = ColoredPointCloud(target_pts, None)
v = TrimeshSceneViewer()
v.add(src_link)
v.add(target_link)
v.show()
import time; time.sleep(1000)
