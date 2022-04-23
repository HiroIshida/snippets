import skrobot
from skrobot.interfaces.ros import PR2ROSRobotInterface  # type: ignore

robot_model = skrobot.models.PR2()
robot_model.init_pose()
robot_interface = PR2ROSRobotInterface(robot_model)
robot_interface.angle_vector(robot_model.angle_vector(), time=10.0, time_scale=1.0)
robot_interface.wait_interpolation()
