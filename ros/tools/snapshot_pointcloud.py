import rospy
import numpy as np
import tf2_ros
import tf2_sensor_msgs
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def callback_pointcloud2(msg, tf_buffer):
    try:
        transform = tf_buffer.lookup_transform('base_footprint', msg.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
        transformed_cloud = tf2_sensor_msgs.do_transform_cloud(msg, transform)
        gen = pc2.read_points(transformed_cloud, skip_nans=True, field_names=("x", "y", "z"))
        points = np.array(list(gen))
        points_filtered = points[points[:, 0] < 2.0]
        np.save('point_cloud.npy', points_filtered)
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(points_filtered[:, 0], points_filtered[:, 1], points_filtered[:, 2], c='b', marker='o')
        # plt.show()
        rospy.signal_shutdown('PointCloud processed')
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        pass

def listener():
    rospy.init_node('pointcloud2_listener', anonymous=True)
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rospy.Subscriber("/kinect_head/depth_registered/half/points", PointCloud2, callback_pointcloud2, tf_buffer)
    rospy.spin()

if __name__ == '__main__':
    listener()
