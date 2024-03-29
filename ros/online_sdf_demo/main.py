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
import trimesh

nodebug = True
with_real_robot = True
with_move_base = True

robot_model = skrobot.models.PR2()

link_list = [
    robot_model.r_shoulder_pan_link, robot_model.r_shoulder_lift_link,
    robot_model.r_upper_arm_roll_link, robot_model.r_elbow_flex_link,
    robot_model.r_forearm_roll_link, robot_model.r_wrist_flex_link,
    robot_model.r_wrist_roll_link]
joint_list = [link.joint for link in link_list]

coll_link_list = [
    robot_model.r_upper_arm_link, robot_model.r_forearm_link,
    robot_model.r_gripper_palm_link, robot_model.r_gripper_r_finger_link,
    robot_model.r_gripper_l_finger_link]

tjoint_angle = robot_model.torso_lift_joint.joint_angle()
robot_model.reset_manip_pose()
robot_model.torso_lift_joint.joint_angle(tjoint_angle)


if nodebug:
    ri = skrobot.interfaces.ros.PR2ROSRobotInterface(robot_model)
    robot_model.head_tilt_joint.joint_angle(0.9)
    ri.angle_vector(robot_model.angle_vector(), time=1.0, time_scale=1.0)
    ri.wait_interpolation()

    robot_model.head_pan_joint.joint_angle(0.5)
    ri.angle_vector(robot_model.angle_vector(), time=1.0, time_scale=1.0)
    ri.wait_interpolation()

    robot_model.head_pan_joint.joint_angle(-0.5)
    ri.angle_vector(robot_model.angle_vector(), time=1.0, time_scale=1.0)
    ri.wait_interpolation()

    robot_model.head_pan_joint.joint_angle(0.0)
    ri.angle_vector(robot_model.angle_vector(), time=1.0, time_scale=1.0)
    ri.wait_interpolation()


    if with_move_base:
        # TODO 
        print("NOTE that, to use this, one needs to remove point cloud on robot")
        ri.go_pos_unsafe(x=0.1, wait=True)

    topic_name = "/voxblox_node/esdf_map_out"
    esdf = EsdfMapClientInterface(0.05, 16, 100.0)
    def callback(msg):
        print("rec")
        esdf.update(msg)
    rospy.Subscriber(topic_name, Layer, callback)
    time.sleep(18.0)
av_start = get_robot_config(robot_model, joint_list, with_base=False)

# solve inverse kinematics to obtain av_goal
joint_angles = np.deg2rad([-60, 74, -70, -120, -20, -30, 180])
set_robot_config(robot_model, joint_list, joint_angles)
target_coords = skrobot.coordinates.Coordinates([0.9, -0.4, 1.0], [0, 0, 0])

rarm_end_coords = skrobot.coordinates.CascadedCoords(
    parent=robot_model.r_gripper_tool_frame,
    name='rarm_end_coords')
robot_model.inverse_kinematics(
    target_coords=target_coords,
    link_list=link_list,
    move_target=robot_model.rarm_end_coords, rotation_axis=False)
av_goal = get_robot_config(robot_model, joint_list, with_base=False)

if nodebug:
    def sdf(X):
        dists = esdf.get_distance(X)
        return dists

    sscc = SweptSphereSdfCollisionChecker(sdf , robot_model)
else:
    sscc = SweptSphereSdfCollisionChecker(lambda X: np.ones(len(X)), robot_model)

for link in coll_link_list:
    sscc.add_collision_link(link)


# motion planning
ts = time.time()
n_waypoint = 15
av_seq = sqp_plan_trajectory(
    sscc, av_start, av_goal, joint_list, n_waypoint,
    safety_margin=0.12, with_base=False)
print("solving time : {0} sec".format(time.time() - ts))

b_min = [0.0, -1.0, 0.0]
b_max = [1.5, 1.0, 1.5]
pts_inside = esdf.debug_points(b_min, b_max)

viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
viewer.add(robot_model)
sscc.add_coll_spheres_to_viewer(viewer)
tpc = trimesh.PointCloud(pts_inside)
pclink = skrobot.model.primitives.PointCloudLink(tpc)
viewer.add(pclink)
viewer.show()

for av in av_seq:
    set_robot_config(robot_model, joint_list, av, with_base=False)
    sscc.update_color()
    viewer.redraw()
    if with_real_robot:
        ri.angle_vector(robot_model.angle_vector(), time=0.5, time_scale=1.0)
        ri.wait_interpolation()
    else:
        time.sleep(2.0)
