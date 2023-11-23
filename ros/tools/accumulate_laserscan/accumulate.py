#!/usr/bin/env python
import matplotlib.pyplot as plt
import tqdm
import pickle
import datetime
import tf2_ros
import tf2_sensor_msgs
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, JointState, LaserScan
import laser_geometry.laser_geometry as lg
import tf
import numpy as np

class LaserScanToPointCloud:
    def __init__(self, n_collect: int = 1000):
        self.pcloud_msg_list = []
        self.joint_state = None
        self.pbar = tqdm.tqdm(total=n_collect, desc="Accumulating Point Clouds")  # Initialize tqdm progress bar
        self.n_collect = n_collect

        self.laser_subscriber = rospy.Subscriber("/tilt_scan_shadow_filtered", PointCloud2, self.callback)
        self.joint_state_subscriber = rospy.Subscriber("/joint_states", JointState, self.callback_joint_state)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.transform_pre = None
        self.quat_history = []
        self.diff_history = []

    def callback(self, pointcloud_msg):
        try:
            transform = self.tf_buffer.lookup_transform('base_footprint', pointcloud_msg.header.frame_id, rospy.Time(0), rospy.Duration(0.01))
            quat_now = np.array([transform.transform.rotation.y, transform.transform.rotation.w])
            if len(self.quat_history) == 0:
                diff_quat = np.inf
            else:
                quat_pre = self.quat_history[-1]
                diff_quat = np.linalg.norm(quat_now - quat_pre)
                self.diff_history.append(diff_quat)
            self.quat_history.append(quat_now)
            if len(self.diff_history) < 10:
                return
            mean_diff_recent = np.mean(self.diff_history[-10:])
            if mean_diff_recent > 0.005:
                rospy.logerr("diff quat too large, skip")
                return

            transformed_cloud = tf2_sensor_msgs.do_transform_cloud(pointcloud_msg, transform)
            self.pcloud_msg_list.append(transformed_cloud)
            self.pbar.update(1)  # Update progress bar
            if len(self.pcloud_msg_list) >= self.n_collect:
                self.pbar.close()  # Close the progress bar
                self.process()
                rospy.signal_shutdown('Processing failed')
                rospy.signal_shutdown('PointCloud processed')
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("TF Error: %s" % str(e))

    def callback_joint_state(self, joint_state_msg):
        self.joint_state = joint_state_msg

    def process(self):
        plt.plot(self.diff_history)
        plt.show()
        rospy.loginfo("Processing point cloud")
        points_list = []
        for pointcloud_msg in self.pcloud_msg_list:
            gen = pc2.read_points(pointcloud_msg, skip_nans=True, field_names=("x", "y", "z"))
            points = np.array(list(gen))
            if points.shape[0] > 0:
                points_list.append(points)
        points_concat = np.vstack(points_list)
        print("processed and obtained {} points".format(points_concat.shape[0]))

        date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        file_name = "pointcloud_" + date_str + ".pkl"
        assert self.joint_state is not None
        with open(file_name, "wb") as f:
            pickle.dump((points_concat, self.joint_state), f)

if __name__ == '__main__':
    rospy.init_node('laser_scan_to_point_cloud')
    l2pc = LaserScanToPointCloud()
    rospy.spin()
