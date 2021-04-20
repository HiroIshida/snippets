import time
import rospy
import numpy as np

from voxblox_msgs.msg import Layer
from voxblox_ros_python import EsdfMapClientInterface

import skrobot
from skrobot.planner.utils import get_robot_config
from skrobot.planner.utils import set_robot_config
from skrobot.planner import sqp_plan_trajectory
from skrobot.planner import SweptSphereSdfCollisionChecker
from skrobot.model.primitives import Box
from utils import *

robot_model = skrobot.models.PR2()
tjoint_angle = robot_model.torso_lift_joint.joint_angle()
robot_model.reset_manip_pose()
robot_model.torso_lift_joint.joint_angle(tjoint_angle)
robot_model.head_tilt_joint.joint_angle(0.8)
ri = skrobot.interfaces.ros.PR2ROSRobotInterface(robot_model)
ri.angle_vector(robot_model.angle_vector(), time=1.0, time_scale=1.0)
ri.wait_interpolation()

topic_name = "/voxblox_node/esdf_map_out"

esdf = EsdfMapClientInterface(0.05, 16, 100.0)
def callback(msg):
    print("rec")
    esdf.update(msg)
rospy.Subscriber(topic_name, Layer, callback)
rospy.spin()
