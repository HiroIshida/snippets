#!/usr/bin/env python
import rospy
import tf
import copy
import numpy as np
from posedetection_msgs.msg import ObjectDetection
from geometry_msgs.msg import Pose, PoseStamped
from particle_filter import ParticleFilter
from kalman_filter import KalmanFilter
from utils import convert

rospy.init_node("pose3d_estimator", anonymous=True)
listener = tf.TransformListener()


def create_posemsg_from_pose3d(pose3d):
    trans_new = [pose3d[0], pose3d[1], 1.126]
    rot_new = tf.transformations.quaternion_from_euler(0.0, pose3d[2], 0.0)
    pose_msg = Pose()

    pose_msg.position.x = trans_new[0]
    pose_msg.position.y = trans_new[1]
    pose_msg.position.z = trans_new[2]

    pose_msg.orientation.x = rot_new[0]
    pose_msg.orientation.y = rot_new[1]
    pose_msg.orientation.z = rot_new[2]
    pose_msg.orientation.w = rot_new[3]
    return pose_msg

class PoseEstimater:
    def __init__(self, N):
        self.N = N
        self.state_estimater = KalmanFilter()
        self.sub = rospy.Subscriber("/kinect_head/rgb/ObjectDetection", 
                ObjectDetection, self.cb_object_detection, queue_size=10)
        self.pub = rospy.Publisher('fridge_pose', PoseStamped, queue_size=10)

    def cb_object_detection(self, msg):
        header = copy.deepcopy(msg.header)
        def convert_pose2tf(pose):
            pos = pose.position
            rot = pose.orientation
            trans = [pos.x, pos.y, pos.z]
            rot = [rot.x, rot.y, rot.z, rot.w]
            return (trans, rot)

        header = msg.header
        assert len(msg.objects) == 1
        obj = msg.objects[0]
        tf_handle_to_kinect = convert_pose2tf(obj.pose)
        tf_kinect_to_map = listener.lookupTransform('/map', '/head_mount_kinect_rgb_optical_frame', rospy.Time(0))

        dist_to_kinect = np.linalg.norm(tf_handle_to_kinect[0])
        print("dist to kinnect is {0}".format(dist_to_kinect))

        tf_handle_to_map = convert(tf_handle_to_kinect, tf_kinect_to_map)
        trans = tf_handle_to_map[0]
        rpy = tf.transformations.euler_from_quaternion(tf_handle_to_map[1])

        state = np.array([trans[0], trans[1], rpy[1]])
        std_x = dist_to_kinect * 0.5
        std_y = dist_to_kinect * 0.5
        std_z = dist_to_kinect * 0.1

        cov = np.diag([std_x**2, std_y**2, std_z**2])
        if self.state_estimater.isInitialized:
            self.state_estimater.update(state, cov)
        else:
            cov_init = cov * 10
            self.state_estimater.initialize(state, cov_init)

        x_est, cov = self.state_estimater.get_current_est(withCov=None)
        print(x_est)
        pose_msg = create_posemsg_from_pose3d(x_est)
        header.frame_id = "/map"
        posestamped_msg = PoseStamped(header = header, pose = pose_msg)
        self.pub.publish(posestamped_msg)

PoseEstimater(5000)
rospy.spin()


