#!/usr/bin/env python  
import rospy
import numpy as np
import math
import tf
import pickle
from sensor_msgs.msg import JointState
import copy

rospy.init_node('teach')
listener = tf.TransformListener()
global angle_vector
angle_vector_ = None

def callback_joint_states(msg):
    D = {}
    for name, position in zip(msg.name, msg.position):
        D[name] = position
    angle_lst = []
    # convert to euslisp angle vector
    pr2_joint_name_lst = [
            "torso_lift_joint", "l_shoulder_pan_joint", "l_shoulder_lift_joint", "l_upper_arm_roll_joint", 
            "l_elbow_flex_joint","l_forearm_roll_joint", "l_wrist_flex_joint", "l_wrist_roll_joint",
            "r_shoulder_pan_joint", "r_shoulder_lift_joint", "r_upper_arm_roll_joint", "r_elbow_flex_joint",
            "r_forearm_roll_joint", "r_wrist_flex_joint", "r_wrist_roll_joint", "head_pan_joint",
            "head_tilt_joint"]
    for name in pr2_joint_name_lst:
        if name == "torso_lift_joint":
            angle_lst.append(D[name] * 1000)
        else:
            angle_lst.append(D[name] * 180/math.pi)
    global angle_vector_
    angle_vector_ = angle_lst

sub_joint_states = rospy.Subscriber('/joint_states', JointState, callback_joint_states)

def get_single_pose():
    print("hit some key")
    string = raw_input()
    if string=='q':
        return None
    
    while True:
        try:
            tf_relative = listener.lookupTransform('/handle', '/r_gripper_tool_frame', rospy.Time(0))
            tf_absolute = listener.lookupTransform('/base_link', '/r_gripper_tool_frame', rospy.Time(0))
            global angle_vector_
            angle_vector = copy.copy(angle_vector_)

            return list(tf_relative), list(tf_absolute), angle_vector
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

if __name__ == '__main__':
    tf_relative_lst = []
    angle_vector_lst = []
    while True:
        try:
            tf_relative, tf_absolute, angle_vector = get_single_pose()
            tf_relative_lst.append(tf_relative)
            angle_vector_lst.append(angle_vector)
        except TypeError:
            print("finish")
            sub_joint_states.unregister()
            break

    with open("tf_raltive_teach.pickle", 'wb') as f:
        data = {'angle_vector': angle_vector, 'tf_relative': tf_relative}
        pickle.dump(data, f)

