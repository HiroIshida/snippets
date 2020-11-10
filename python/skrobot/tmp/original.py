import skrobot
robot_model = skrobot.models.urdf.RobotModelFromURDF(urdf_file=skrobot.data.pr2_urdfpath())
viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
viewer.add(robot_model)
viewer.show()

robot_model.l_elbow_flex_joint.joint_angle(-0.5)

robot_model.torso_lift_joint.joint_angle(0.05)
robot_model.l_shoulder_pan_joint.joint_angle(60 * 3.14/180.0)
robot_model.l_shoulder_lift_joint.joint_angle(74 * 3.14/180.0)
robot_model.l_upper_arm_roll_joint.joint_angle(70* 3.14/180.0)
robot_model.l_elbow_flex_joint.joint_angle(-120 * 3.14/180.0)
robot_model.l_forearm_roll_joint.joint_angle(20 * 3.14/180.0)
robot_model.l_wrist_flex_joint.joint_angle(-30 * 3.14/180.0)
robot_model.l_wrist_roll_joint.joint_angle(180 * 3.14/180.0)
robot_model.r_shoulder_pan_joint.joint_angle(-60 * 3.14/180.0)
robot_model.r_shoulder_lift_joint.joint_angle(74 * 3.14/180.0)
robot_model.r_upper_arm_roll_joint.joint_angle(-70 * 3.14/180.0)
robot_model.r_elbow_flex_joint.joint_angle(-120 * 3.14/180.0)
robot_model.r_forearm_roll_joint.joint_angle(-20 * 3.14/180.0)
robot_model.r_wrist_flex_joint.joint_angle(-30 * 3.14/180.0)
robot_model.r_wrist_roll_joint.joint_angle(180 * 3.14/180.0)
robot_model.head_pan_joint.joint_angle(0)
robot_model.head_tilt_joint.joint_angle(0)

rarm_end_coords = skrobot.coordinates.CascadedCoords(
            parent=robot_model.r_gripper_tool_frame,
            name='rarm_end_coords')
move_target = rarm_end_coords
link_list = [
    robot_model.r_shoulder_pan_link,
    robot_model.r_shoulder_lift_link,
    robot_model.r_upper_arm_roll_link,
    robot_model.r_elbow_flex_link,
    robot_model.r_forearm_roll_link,
    robot_model.r_wrist_flex_link,
    robot_model.r_wrist_roll_link]

target_coords = skrobot.coordinates.Coordinates([0.5, -0.3, 0.7], [0, 0, 0])
robot_model.inverse_kinematics(
        target_coords, link_list=link_list, move_target=move_target)
