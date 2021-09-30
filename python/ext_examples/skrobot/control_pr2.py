import skrobot

robot_model = skrobot.models.PR2()
robot_model.reset_manip_pose()
robot_model.reset_pose()

viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
viewer.add(robot_model)

ri = skrobot.interfaces.ros.PR2ROSRobotInterface(robot_model)
# time_scale must be 1.0
ri.angle_vector(robot_model.angle_vector(), time=1.0, time_scale=1.0)
