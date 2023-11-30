import numpy as np
import tf2_sensor_msgs
import tf2_ros
from ros_numpy.point_cloud2 import get_xyz_points, pointcloud2_to_array, split_rgb_field
import rospy
import message_filters
from std_msgs.msg import Int32, Float32
from jsk_recognition_msgs.msg import ClusterPointIndices
from sensor_msgs.msg import PointCloud2

def callback(cluster_indices: ClusterPointIndices, pcloud: PointCloud2, tfbuffer):
    if len(cluster_indices.cluster_indices) > 1:
        rospy.logwarn("more than 1 cluster detected. skip this frame.")
        return

    # I dont know why but without transformation here, ros_numpy's conversion fails 
    transform = tf_buffer.lookup_transform('base_footprint', pcloud.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
    transformed_cloud = tf2_sensor_msgs.do_transform_cloud(pcloud, transform)
    arr = pointcloud2_to_array(transformed_cloud)
    xyz = get_xyz_points(arr, remove_nans=False)
    detic_indices = np.array(cluster_indices.cluster_indices[0].indices, dtype=np.int32)

    finite_indices = np.sum(np.isnan(xyz), axis=1) == 0
    indices = np.intersect1d(detic_indices, np.where(finite_indices)[0])

    xyz = xyz[indices]
    rgb = split_rgb_field(arr)
    r = rgb["r"][indices]
    g = rgb["g"][indices]
    b = rgb["b"][indices]
    xyzrgb = np.hstack((xyz, np.vstack((r, g, b)).T))

    # save xyzrgb as npy
    np.save("/tmp/cup_fridge.npy", xyzrgb)

    from skrobot.viewers import TrimeshSceneViewer
    from skrobot.model.primitives import PointCloudLink
    v = TrimeshSceneViewer()
    plink = PointCloudLink(xyzrgb[:, :3], xyzrgb[:, 3:])
    v.add(plink)
    v.show()
    import time; time.sleep(1000)


cluster_indices_sub = message_filters.Subscriber("/docker/mugcup/detic_segmentor/indices", ClusterPointIndices)
cloud_sub = message_filters.Subscriber("/docker/tf_transform/output", PointCloud2)

rospy.init_node("tmp")

tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)
ts = message_filters.ApproximateTimeSynchronizer([cluster_indices_sub, cloud_sub], 50, 0.5, allow_headerless=True)
ts.registerCallback(callback, tf_buffer)
rospy.spin()
