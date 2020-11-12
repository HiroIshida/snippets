import skrobot
import pybullet
import pybullet as pb
import numpy as np
import time 
import copy
from utils import sdf_box
from skrobot.interfaces import PybulletRobotInterface

def sdf_box(b, c):
    mat = None
    closure_member = {"rotmat": mat}

    def sdf(X):
        rotmat = closure_member["rotmat"]
        if rotmat is not None:
            X = X.dot(closure_member["rotmat"]) # equivalent to inverse (transpose)
        n_pts = X.shape[0]
        dim = X.shape[1]
        center = np.array(c).reshape(1, dim)
        center_copied = np.repeat(center, n_pts, axis=0)
        P = X - center_copied
        Q = np.abs(P) - np.repeat(np.array(b).reshape(1, dim), n_pts, axis=0)
        left__ = np.array([Q, np.zeros((n_pts, dim))])
        left_ = np.max(left__, axis = 0)
        left = np.sqrt(np.sum(left_**2, axis=1))
        right_ = np.max(Q, axis=1)
        right = np.min(np.array([right_, np.zeros(n_pts)]), axis=0)
        return left + right 
    return sdf

try:
    robot_model
except:
    robot_model = skrobot.models.urdf.RobotModelFromURDF(urdf_file=skrobot.data.pr2_urdfpath())
    robot_model.init_pose()
    client_id = pybullet.connect(pybullet.GUI)
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0)
    viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
    interface = PybulletRobotInterface(robot_model, connect=client_id)
    interface.angle_vector(robot_model.angle_vector())

    link_idx_table = {}
    for link_idx in range(len(robot_model.link_list)):
        name = robot_model.link_list[link_idx].name
        link_idx_table[name] = link_idx

    rarm_end_coords = skrobot.coordinates.CascadedCoords(
            parent=robot_model.r_gripper_tool_frame) 
    move_target = rarm_end_coords

    forarm_coords = skrobot.coordinates.CascadedCoords(
            parent=robot_model.r_forearm_link) 

def create_box(center, b):
    quat = [0, 0, 0, 1]
    vis_id = pb.createVisualShape(pb.GEOM_BOX, halfExtents=b, rgbaColor=[0.0, 1.0, 0, 0.7], physicsClientId=client_id)
    pb.createMultiBody(basePosition=center, baseOrientation=quat, baseVisualShapeIndex=vis_id)
    sdf = sdf_box(b, center) 
    return sdf

link_names = ["r_shoulder_pan_link", "r_shoulder_lift_link", \
        "r_upper_arm_roll_link", "r_elbow_flex_link", \
        "r_forearm_roll_link", "r_wrist_flex_link", "r_wrist_roll_link"]

link_list = [robot_model.link_list[link_idx_table[name]] for name in link_names]
joint_list = [link.joint for link in link_list]

set_joint_angles = lambda av: [j.joint_angle(a) for j, a in zip(joint_list, av)]
get_joint_angles = lambda : np.array([j.joint_angle() for j in joint_list])

set_joint_angles([0]*7)
interface.angle_vector(robot_model.angle_vector())

target_coords = skrobot.coordinates.Coordinates([0.5, 0.5, 0.8], [0, 0, 0])
res = robot_model.inverse_kinematics(
        target_coords, link_list=link_list, move_target=move_target, rotation_axis=False)
av_init = get_joint_angles()
set_joint_angles(av_init)
print("setting")
time.sleep(1.0)
interface.angle_vector(robot_model.angle_vector())

target_coords = skrobot.coordinates.Coordinates([0.7, -0.7, 1.0], [0, 0, 0])
res = robot_model.inverse_kinematics(
        target_coords, link_list=link_list, move_target=move_target, rotation_axis=False)
av_target = get_joint_angles()
n_wp = 8
step = (av_target - av_init)/(n_wp - 1)
init_trajectory = np.array([av_init + i * step for i in range(n_wp)])

target_coords = skrobot.coordinates.Coordinates([0.7, -0.7, 1.0], [0, 0, 0])
sdf = create_box([0.9, -0.2, 0.9], [0.4, 0.25, 0.3])

traj = robot_model.plan_trajectory(
        target_coords, 
        10, 
        link_list, 
        rarm_end_coords,
        [rarm_end_coords, forarm_coords], 
        sdf,
        initial_trajectory=init_trajectory
        )

time.sleep(1.0)
print("show trajectory")
for av in traj:
    set_joint_angles(av)
    interface.angle_vector(robot_model.angle_vector())
    time.sleep(0.5)
