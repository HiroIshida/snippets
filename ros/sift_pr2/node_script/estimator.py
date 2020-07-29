#!/usr/bin/env python
import rospy
import tf
import numpy as np
from posedetection_msgs.msg import ObjectDetection
from particle_filter import ParticleFilter

rospy.init_node("pose3d_estimator", anonymous=True)
listener = tf.TransformListener()

def qv_mult(q1, v1_):
    length = np.linalg.norm(v1_)
    v1 = v1_/length
    v1 = tf.transformations.unit_vector(v1)
    q2 = list(v1)
    q2.append(0.0)
    v_converted = tf.transformations.quaternion_multiply(
        tf.transformations.quaternion_multiply(q1, q2), 
        tf.transformations.quaternion_conjugate(q1)
    )[:3]
    return v_converted * length

def convert(tf_12, tf_23):
    tran_12, rot_12 = [np.array(e) for e in tf_12]
    tran_23, rot_23 = [np.array(e) for e in tf_23]

    rot_13 = tf.transformations.quaternion_multiply(rot_12, rot_23)
    tran_13 = tran_23 + qv_mult(rot_23, tran_12)
    return list(tran_13), list(rot_13)

class PoseEstimater:
    def __init__(self, N):
        self.N = N
        self.pf = ParticleFilter(N)
        self.sub = rospy.Subscriber("/kinect_head/rgb/ObjectDetection", 
                ObjectDetection, self.cb_object_detection, queue_size=10)

    def cb_object_detection(self, msg):
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
        tf_handle_to_map = convert(tf_handle_to_kinect, tf_kinect_to_map)
        trans = tf_handle_to_map[0]
        rpy = tf.transformations.euler_from_quaternion(tf_handle_to_map[1])

        state = np.array([trans[0], trans[1], rpy[1]])
        cov = np.diag([0.2**2, 0.2**2, 0.1**2])
        if self.pf.X is None:
            cov_init = cov * 10
            ptcls = np.random.multivariate_normal(state, cov, self.N)
            self.pf.initialize(ptcls)
        else:
            self.pf.update(state, cov)
        x_est, cov = self.pf.get_current_est()
        print(np.diag(cov))


PoseEstimater(1000)
rospy.spin()


