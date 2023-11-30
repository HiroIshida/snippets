from scipy.optimize import minimize
from skimage import io, color
from cmaes import CMA
from scipy.spatial import KDTree
import numpy as np
from skrobot.coordinates import Coordinates
from skrobot.model.link import Link
from skrobot.viewers import TrimeshSceneViewer
from trimesh import PointCloud


def rgb_to_hsv(rgb):
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    rangec = maxc - minc

    h = np.zeros_like(maxc)
    s = np.zeros_like(maxc)
    v = maxc

    nonzero_rangec = rangec != 0
    s[nonzero_rangec] = rangec[nonzero_rangec] / maxc[nonzero_rangec]

    rc = np.where(nonzero_rangec, (maxc - rgb[..., 0]) / rangec, 0)
    gc = np.where(nonzero_rangec, (maxc - rgb[..., 1]) / rangec, 0)
    bc = np.where(nonzero_rangec, (maxc - rgb[..., 2]) / rangec, 0)

    maxc_is_r = np.isclose(rgb[..., 0], maxc)
    maxc_is_g = np.isclose(rgb[..., 1], maxc)
    maxc_is_b = np.isclose(rgb[..., 2], maxc)

    h[maxc_is_r] = bc[maxc_is_r] - gc[maxc_is_r]
    h[maxc_is_g] = 2.0 + rc[maxc_is_g] - bc[maxc_is_g]
    h[maxc_is_b] = 4.0 + gc[maxc_is_b] - rc[maxc_is_b]

    h = (h / 6.0) % 1.0

    return np.stack((h, s, v), axis=-1)


class ColoredPointCloud(Link):
    def __init__(self, pts, colors, name=None):
        super().__init__(name=name)
        self.pts = pts
        self.colors = colors
        self._visual_mesh = PointCloud(pts, colors)


src = np.load('point_cloud_colored1_filtered.npy')
src_pts = src[:, :3]
src_colors = src[:, 3:]

# target = np.load('point_cloud_colored2_filtered.npy')
target = np.load('cup_fridge.npy')
# target = src.copy()
target_pts = target[:, :3]
target_colors = target[:, 3:]

tree = KDTree(target_pts)

def get_co(planer_pose: np.ndarray):
    x, y, z, roll, pitch, yaw = planer_pose
    co = Coordinates(pos=[x, y, z], rot=[yaw, pitch, roll])
    return co

def loss(planer_pose: np.ndarray):
    co = get_co(planer_pose)
    src_pts_converted = co.transform_vector(src_pts)
    _, indices = tree.query(src_pts_converted)
    target_pts_nearest = target_pts[indices]
    assert src_pts_converted.shape == target_pts_nearest.shape

    loss_position = np.sum(np.linalg.norm(src_pts_converted - target_pts_nearest, axis=1))

    src_lab = rgb_to_hsv(src_colors)[:, 0]
    target_lab = rgb_to_hsv(target_colors[indices])[:, 0]
    loss_color = np.sum(np.linalg.norm(src_lab - target_lab))
    # loss_color = np.sum(np.linalg.norm(src_lab - target_lab, axis=1))
    print(loss_position, loss_color * 1e-2)
    return loss_position + loss_color * 1e-2


# initial guess
target_center = np.mean(target_pts, axis=0)
src_center = np.mean(src_pts, axis=0)
guess = np.hstack([target_center - src_center, np.zeros(3)])

# optimize
use_scipy = False
if use_scipy:
    res = minimize(loss, guess, method='nelder-mead', options={'xatol': 1e-8, 'disp': True, "maxiter": 10000})
    print(res.x)
    param_best = res.x
else:
    optimizer = CMA(mean=guess, sigma=2.0)# 
    for _ in range(1000):
        dataset = []
        for _ in range(optimizer.population_size):
            param = optimizer.ask()
            val = loss(param)
            dataset.append((param, val))
        optimizer.tell(dataset)

        val_list = [val for param, val in dataset]
        idx_best = np.argmin(val_list)
        param_best = dataset[idx_best][0]
        print(param_best, val_list[idx_best])

        if optimizer.should_stop():
            break


co = get_co(param_best)
src_pts_converted = co.transform_vector(src_pts)

src_link = ColoredPointCloud(src_pts_converted, src_colors)
target_link = ColoredPointCloud(target_pts, target_colors)
v = TrimeshSceneViewer()
v.add(src_link)
v.add(target_link)
v.show()
import time; time.sleep(1000)
