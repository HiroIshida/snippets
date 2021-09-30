import skrobot
try:
    robot_model
except:
    robot_model = skrobot.models.urdf.RobotModelFromURDF(urdf_file=skrobot.data.pr2_urdfpath())
    robot_model.init_pose()
    viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
    viewer.add(robot_model)
    viewer.show()

coef = 3.1415 /180.0
robot_model.torso_lift_joint.joint_angle(0.05*coef)
robot_model.r_shoulder_pan_joint.joint_angle(-60*coef)
robot_model.r_shoulder_lift_joint.joint_angle(74*coef)
robot_model.r_upper_arm_roll_joint.joint_angle(-70*coef)
robot_model.r_elbow_flex_joint.joint_angle(-120*coef)
robot_model.r_forearm_roll_joint.joint_angle(-20*coef)
robot_model.r_wrist_flex_joint.joint_angle(-30*coef)
robot_model.r_wrist_roll_joint.joint_angle(180*coef)
robot_model.head_pan_joint.joint_angle(0)
robot_model.head_tilt_joint.joint_angle(0)



rarm_end_coords = skrobot.coordinates.CascadedCoords(
        parent=robot_model.r_gripper_tool_frame, 
        name='rarm_end_coords')

move_target = rarm_end_coords
link_list = [robot_model.r_shoulder_pan_link,
        robot_model.r_shoulder_lift_link,
        robot_model.r_upper_arm_roll_link,
        robot_model.r_elbow_flex_link,
        robot_model.r_forearm_roll_link,
        robot_model.r_wrist_flex_link,
        robot_model.r_wrist_roll_link]
#target_coords = rarm_end_coords.copy_worldcoords()
#target_coords.translate((-0.1, 0, 0), 'local')
target_coords = skrobot.coordinates.Coordinates([0.5, -0.3, 0.7], [0, 0, 0])
robot_model.inverse_kinematics(
        target_coords, link_list=link_list, move_target=move_target)

