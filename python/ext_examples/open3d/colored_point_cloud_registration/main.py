import open3d as o3d
import numpy as np
from trimesh import PointCloud
from skrobot.model.link import Link
from skrobot.viewers import TrimeshSceneViewer

colored_points1 = np.load('point_cloud_colored1_filtered.npy')
colored_points2 = np.load('point_cloud_colored2_filtered.npy')

points1, colors1 = colored_points1[:, :3], colored_points1[:, 3:]
points2, colors2 = colored_points2[:, :3], colored_points2[:, 3:]

source = o3d.geometry.PointCloud()
source.points = o3d.utility.Vector3dVector(points1)
source.colors = o3d.utility.Vector3dVector(colors1 / 255.0)

target = o3d.geometry.PointCloud()
target.points = o3d.utility.Vector3dVector(points2)
target.colors = o3d.utility.Vector3dVector(colors2 / 255.0)

radius = 0.002

source_down = source.voxel_down_sample(radius)
target_down = target.voxel_down_sample(radius)
source_down.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
target_down.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))


current_transformation = np.identity(4)

# o3d.visualization.draw_geometries([source_down, target_down])
criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-1, relative_rmse=1e-1, max_iteration=30)

result_icp = o3d.pipelines.registration.registration_colored_icp(
    source_down, target_down, radius, current_transformation,
    o3d.pipelines.registration.TransformationEstimationForColoredICP(),
    criteria)
current_transformation = result_icp.transformation
print(result_icp, "\n")
print(current_transformation)

source.transform(current_transformation)
o3d.visualization.draw_geometries([source, target])
