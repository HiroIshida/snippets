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

def test():
    while True:
        try:
            tf1 = listener.lookupTransform('/base_link', '/oven', rospy.Time(0))
            tf2 = listener.lookupTransform('/oven', '/handle', rospy.Time(0))
            tf3 = listener.lookupTransform('/base_link', '/handle', rospy.Time(0))
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
    return list(tran_bh), list(tran_bo)



if __name__=='__main__':
    tf1, tf2, tf3 = test()
    #tran, rot = convert(tf1, tf2)

    trans1, rot1 = [np.array(e) for e in tf1]
    trans2, rot2 = [np.array(e) for e in tf2]
    trans3, rot3 = [np.array(e) for e in tf3]

    tmp = tf.transformations.quaternion_multiply(rot1, rot2) 
    print(tmp - rot3)

    v = qv_mult(rot1, trans2)
    print(trans1 + qv_mult(rot1, trans2))
    print(trans3)
