import numpy as np
from trimesh import PointCloud
from skrobot.model.link import Link
from skrobot.viewers import TrimeshSceneViewer

colored_points = np.load('point_cloud_colored1.npy')
points = colored_points[:, :3]
colors = colored_points[:, 3:].astype(np.uint8)
colors = np.hstack((colors, np.ones((colors.shape[0], 1), dtype=np.uint8) * 255))

indices = np.all((
    points[:, 0] < 0.6,
    points[:, 1] > -0.1,
    points[:, 1] < 0.1,
    points[:, 2] > 0.73,
    points[:, 2] < 1.2,
    ), axis=0)

colored_points2 = np.load('point_cloud_colored2.npy')
points2 = colored_points2[:, :3]
colors2 = colored_points2[:, 3:].astype(np.uint8)
colors2 = np.hstack((colors2, np.ones((colors2.shape[0], 1), dtype=np.uint8) * 255))

indices2 = np.all((
    points2[:, 0] < 0.6,
    points2[:, 1] > -0.1,
    points2[:, 1] < 0.1,
    points2[:, 2] > 0.73,
    points2[:, 2] < 1.2,
    ), axis=0)


# save filtered point cloud as npy
np.save('point_cloud_colored1_filtered.npy', colored_points[indices])
np.save('point_cloud_colored2_filtered.npy', colored_points2[indices2])

pcloud = PointCloud(points[indices], colors=colors[indices])
pcloud2 = PointCloud(points2[indices2], colors=colors2[indices2])
link = Link()
link._visual_mesh = pcloud
link2 = Link()
link2._visual_mesh = pcloud2
v = TrimeshSceneViewer()
v.add(link)
v.add(link2)
v.show()
import time; time.sleep(1000)
