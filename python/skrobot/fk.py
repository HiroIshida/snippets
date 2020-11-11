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


def compute_jacobian_skrobot(av0):
    set_joint_angles(av0)

    base_link = robot_model.link_list[0]
    J = robot_model.calc_jacobian_from_link_list([move_target], link_list, transform_coords=base_link)
    return J

def compute_jacobain_naively(av0): # by finite diff
    set_joint_angles(av0)
    current_coords = rarm_end_coords.copy_worldcoords()
    pos0 = current_coords._translation
    J = np.zeros((3, 7))

    eps = -1e-5
    for idx in range(7):
        av1 = copy.copy(av0)
        av1[idx] += eps 
        set_joint_angles(av1)

        tweaked_coords = rarm_end_coords.copy_worldcoords()
        pos1 = tweaked_coords._translation
        pos_diff = (pos1 - pos0)/eps
        J[:, idx] = pos_diff

    set_joint_angles(av0) # recover to the original state
    return J

def compare_jacobian(av0):
    J_mine = compute_jacobain_naively(av0)
    J_skrobot = compute_jacobian_skrobot(av0)
    diff = J_mine - J_skrobot
    print(np.round(diff, 3))

#av = [0.1]+[0]*6
av = [-0.3]*7

compare_jacobian(av) 

set_joint_angles([0.1]+[0]*6)
current_coords = rarm_end_coords.copy_worldcoords()
pos0 = current_coords._translation

