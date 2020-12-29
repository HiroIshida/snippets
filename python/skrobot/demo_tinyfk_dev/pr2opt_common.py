import skrobot

def pr2_init():
    robot_model = skrobot.models.PR2()
    robot_model.reset_manip_pose()
    return robot_model

def rarm_joint_list(robot_model):
    link_list = [
        robot_model.r_shoulder_pan_link, robot_model.r_shoulder_lift_link,
        robot_model.r_upper_arm_roll_link, robot_model.r_elbow_flex_link,
        robot_model.r_forearm_roll_link, robot_model.r_wrist_flex_link,
        robot_model.r_wrist_roll_link,
        robot_model.l_shoulder_pan_link, robot_model.l_shoulder_lift_link,
        robot_model.l_upper_arm_roll_link, robot_model.l_elbow_flex_link,
        robot_model.l_forearm_roll_link, robot_model.l_wrist_flex_link,
        robot_model.l_wrist_roll_link]
    joint_list = [link.joint for link in link_list]
    return joint_list

def rarm_coll_link_list(robot_model):
    coll_link_list = [
        robot_model.r_upper_arm_link, robot_model.r_forearm_link,
        robot_model.r_gripper_palm_link, robot_model.r_gripper_r_finger_link,
        robot_model.r_gripper_l_finger_link,
        robot_model.l_upper_arm_link, robot_model.l_forearm_link,
        robot_model.l_gripper_palm_link, robot_model.l_gripper_r_finger_link,
        robot_model.l_gripper_l_finger_link
        ]
    return coll_link_list
