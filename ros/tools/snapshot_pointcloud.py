import time
import argparse
import rospy
import numpy as np
import tf2_ros
import tf2_sensor_msgs
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from trimesh import PointCloud
from skrobot.model.link import Link
from skrobot.viewers import TrimeshSceneViewer
from ros_numpy.point_cloud2 import get_xyz_points, pointcloud2_to_array, split_rgb_field


def callback_pointcloud2(msg, tf_buffer):
    try:
        transform = tf_buffer.lookup_transform('base_footprint', msg.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
        transformed_cloud = tf2_sensor_msgs.do_transform_cloud(msg, transform)
        gen = pc2.read_points(transformed_cloud, skip_nans=True, field_names=("x", "y", "z"))
        points = np.array(list(gen))
        points_filtered = points[points[:, 0] < 2.0]
        np.save('point_cloud.npy', points_filtered)
        rospy.signal_shutdown('PointCloud processed')
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        pass


def callback_pointcloud_colored(msg, tf_buffer):
    try:
        ts = time.time()
        transform = tf_buffer.lookup_transform('base_footprint', msg.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
        transformed_cloud = tf2_sensor_msgs.do_transform_cloud(msg, transform)
        arr = pointcloud2_to_array(transformed_cloud)
        rgb = split_rgb_field(arr)
        xyz = get_xyz_points(arr, remove_nans=False)  # nan filter by myself
        finite_indices = np.sum(np.isnan(xyz), axis=1) == 0
        xyz_finite = xyz[finite_indices]
        r_finite = rgb["r"][finite_indices]
        g_finite = rgb["g"][finite_indices]
        b_finite = rgb["b"][finite_indices]
        xyzrgb = np.hstack((xyz_finite, np.vstack((r_finite, g_finite, b_finite)).T))
        print('time: ', time.time() - ts)
        np.save('point_cloud_colored.npy', xyzrgb)
        rospy.signal_shutdown('PointCloud processed')
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        pass


def listener(save_rgb):
    rospy.init_node('pointcloud2_listener', anonymous=True)
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    callback = callback_pointcloud_colored if save_rgb else callback_pointcloud2
    rospy.Subscriber("/kinect_head/depth_registered/half/points", PointCloud2, callback, tf_buffer)
    rospy.spin()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    if args.test:
        if args.rgb:
            colored_points = np.load('point_cloud_colored.npy')
            points = colored_points[:, :3]
            colors = colored_points[:, 3:].astype(np.uint8)
            colors = np.hstack((colors, np.ones((colors.shape[0], 1), dtype=np.uint8) * 140))
        else:
            points = np.load('point_cloud.npy')
            colors = None
        pcloud = PointCloud(points, colors=colors)
        link = Link()
        link._visual_mesh = pcloud
        v = TrimeshSceneViewer()
        v.add(link)
        v.show()
        import time; time.sleep(1000)
    else:
        listener(args.rgb)
