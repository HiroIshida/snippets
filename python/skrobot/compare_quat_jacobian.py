import skrobot
import numpy as np
import copy

try:
    robot_model
except:
    robot_model = skrobot.models.urdf.RobotModelFromURDF(urdf_file=skrobot.data.pr2_urdfpath())
    robot_model.init_pose()
    viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
    viewer.add(robot_model)
    viewer.show()

    link_idx_table = {}
    for link_idx in range(len(robot_model.link_list)):
        name = robot_model.link_list[link_idx].name
        link_idx_table[name] = link_idx

    rarm_end_coords = skrobot.coordinates.CascadedCoords(
            parent=robot_model.r_gripper_tool_frame, 
            name='rarm_end_coords')
    move_target = rarm_end_coords

    link_names = ["r_shoulder_pan_link", "r_shoulder_lift_link", \
            "r_upper_arm_roll_link", "r_elbow_flex_link", \
            "r_forearm_roll_link", "r_wrist_flex_link", "r_wrist_roll_link"]
    link_list = [robot_model.link_list[link_idx_table[name]] for name in link_names]
    joint_list = [link.joint for link in link_list]

set_joint_angles = lambda av: [j.joint_angle(a) for j, a in zip(joint_list, av)]

from math import *
def pimat(r, p, y): 
    a1 = -r
    a2 = -p
    a3 = -y
    """
    mat = np.array([
        [cos(a3)/cos(a1), -sin(a3)/cos(a2), 0.0],
        [sin(a3), cos(a3), 0.0],
        [-cos(a3)*sin(a2)/cos(a2), sin(a3)*sin(a2)/cos(a2), 1.0]
        ])
    """

    mat = np.array([
        [0, sin(a3)/cos(a2), cos(a3)/cos(a2)],
        [0, cos(a3), -sin(a3)],
        [1.0, sin(a3)*sin(a2)/cos(a3), cos(a3)*sin(a2)/cos(a2)]])
    return mat

def compute_jacobian_skrobot(av0, mat, rotalso=False):
    set_joint_angles(av0)

    base_link = robot_model.link_list[0]
    J = robot_model.calc_jacobian_from_link_list([move_target], link_list, transform_coords=base_link, rotation_axis=rotalso)
    J[3:, :] = mat.dot(J[3:, :])
    return J

from skrobot.coordinates.math import rpy_angle


def compute_jacobain_naively(av0): # by finite diff

    def get_rpy(coords):
        co = coords.copy_worldcoords()
        return co.quaternion

    set_joint_angles(av0)

    pos0 = rarm_end_coords.worldpos()
    rot0 = get_rpy(rarm_end_coords)

    J = np.zeros((7, 7))

    eps = -1e-8
    for idx in range(7):
        av1 = copy.copy(av0)
        av1[idx] += eps 
        set_joint_angles(av1)

        pos1 = rarm_end_coords.worldpos()
        rot1 = get_rpy(rarm_end_coords)
        pos_diff = (pos1 - pos0)/eps
        rot_diff = (rot1 - rot0)/eps
        J[:3, idx] = pos_diff
        J[3:, idx] = rot_diff

    set_joint_angles(av0) # recover to the original state
    return J


def quaternion_relation_mat(coords):
    q = coords.quaternion
    q1, q2, q3, q4 = q
    mat = np.array([
        [-q2, -q3, -q4],
        [q1, q4, -q3],
        [-q4, q1, q2],
        [q3, -q2, q1],
        ])
    return mat * 0.5

av0 = [-0.2]*7 
set_joint_angles(av0)
M = quaternion_relation_mat(rarm_end_coords.copy_worldcoords())

J_mine_rot = compute_jacobain_naively(av0)[3:, :]
J_skrobot_rot = compute_jacobian_skrobot(av0, np.eye(3), rotalso=True)[3:, :]

print(np.round(J_mine_rot - M.dot(J_skrobot_rot), 2))


