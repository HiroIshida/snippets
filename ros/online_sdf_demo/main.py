import skrobot

robot_model = skrobot.models.PR2()
ri = skrobot.interfaces.ros.PR2ROSRobotInterface(robot_model)
robot_model.reset_manip_pose()

robot_model.head_tilt_joint.joint_angle(0.8)
ri.angle_vector(robot_model.angle_vector(), time=1.0, time_scale=1.0)
ri.wait_interpolation()
