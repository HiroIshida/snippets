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

# set initial joints
set_joint_angles = lambda av: [j.joint_angle(a) for j, a in zip(joint_list, av)]

av0 = [-0.01]*7
set_joint_angles(av0)
current_coords = rarm_end_coords.copy_worldcoords()
pos0 = current_coords._translation
rot0 = np.array(skrobot.coordinates.quaternion2rpy(current_coords._q)[0])

J_real = robot_model.calc_jacobian_from_link_list([move_target], link_list, rotation_axis=[True])

def compute_jacobain_naively(av0):
    J = np.zeros((6, 7))
    eps = -1e-5
    for idx in range(7):
        av1 = copy.copy(av0)
        av1[idx] += eps 
        set_joint_angles(av1)

        tweaked_coords = rarm_end_coords.copy_worldcoords()
        pos1 = tweaked_coords._translation
        rot1 = np.array(skrobot.coordinates.quaternion2rpy(tweaked_coords._q)[0])

        pos_diff = (pos1 - pos0)/eps
        rot_diff = (rot1 - rot0)/eps
        J[:3, idx] = pos_diff
        J[3:, idx] = rot_diff
    return J
J_mine = compute_jacobain_naively(av0)
diff = J_mine - J_real
print(np.round(diff, 2))
