import numpy as np
from numpy.linalg import norm
import pickle
import rospy
import json
import tf
from geometry_msgs.msg import Quaternion


rospy.init_node('commander')
listener = tf.TransformListener()
# https://answers.ros.org/question/196149/how-to-rotate-vector-by-quaternion-in-python/
# http://docs.ros.org/jade/api/tf/html/python/transformations.html

def test():
    while True:
        try:
            tf1 = listener.lookupTransform('/base_link', '/l_shoulder_pan_link', rospy.Time(0))
            tf2 = listener.lookupTransform('/l_shoulder_pan_link', '/l_upper_arm_link', rospy.Time(0))
            tf3 = listener.lookupTransform('/base_link', '/l_upper_arm_link', rospy.Time(0))
            return tf1, tf2, tf3
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue


def convert(tf_body_to_oven, tf_oven_to_handle):

    def qv_mult(q1, v1_):
        length = norm(v1_)
        v1 = v1_/length
        v1 = tf.transformations.unit_vector(v1)
        q2 = list(v1)
        q2.append(0.0)
        v_converted = tf.transformations.quaternion_multiply(
            tf.transformations.quaternion_multiply(q1, q2), 
            tf.transformations.quaternion_conjugate(q1)
        )[:3]
        return v_converted * length

    tran_bo, rot_bo = [np.array(e) for e in tf_body_to_oven]
    tran_oh, rot_oh = [np.array(e) for e in tf_oven_to_handle]

    rot_bh = tf.transformations.quaternion_multiply(rot_bo, rot_oh)
    tran_bh = tran_bo + qv_mult(rot_bo, tran_oh)
    return list(tran_bh), list(rot_bh)

if __name__=='__main__':
    tf1, tf2, tf3 = test()
    tran, rot = convert(tf1, tf2)
    tran3, rot3 = tf3

