#!/usr/bin/env python
import silverbullet as sb
import pybullet as pb
import pybullet
import time
import rospy
from geometry_msgs.msg import Pose, PoseStamped
import tf
from utils import convert
from silverbullet.ros_interface import PR2ROSRobotInterface

def convert_posemsg_to_transform(msg):
    pose = msg.pose
    pos = pose.position
    ori = pose.orientation

    trans = [pos.x, pos.y, pos.z]
    rot = [ori.x, ori.y, ori.z, ori.w] # quat
    return [trans, rot]

rospy.init_node("feedback_robot_controller", anonymous=True)
listener = tf.TransformListener()

CLIENT = pybullet.connect(pybullet.GUI)
pb.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0, physicsClientId=CLIENT)
pb.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS, 0, physicsClientId=CLIENT)
pb.configureDebugVisualizer(pybullet.COV_ENABLE_TINY_RENDERER, 0, physicsClientId=CLIENT)
robot = sb.robot_models.PR2(CLIENT)
ri = PR2ROSRobotInterface(robot)



def cb_fridge_pose(msg):
    tf_handle_to_map = convert_posemsg_to_transform(msg)
    tf_map_to_base = listener.lookupTransform("/base_link", "/map", rospy.Time(0))
    tf_handle_to_base = convert(tf_handle_to_map, tf_map_to_base)

    trans, quat = tf_handle_to_base
    rpy = tf.transformations.euler_from_quaternion(quat)
    robot.solve_ik(trans, rpy, group_names=('larm'))
    ri.angle_vector(robot.get_angle_vector(), time=0.2)
    print(trans)

sub = rospy.Subscriber("fridge_pose", PoseStamped, cb_fridge_pose, queue_size=10)
rospy.spin()

"""
q_new = robot.solve_ik(x_target, group_names=('rarm', 'torso'))
x_target = [0.3, 0.5, 1.7]
q_new = robot.solve_ik(x_target, group_names=('larm'))
"""

