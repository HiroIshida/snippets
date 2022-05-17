import math
import skrobot
from skrobot.model import Joint
from skrobot.interfaces.ros import PR2ROSRobotInterface  # type: ignore

robot_model = skrobot.models.PR2()
ri = PR2ROSRobotInterface(robot_model)

while True:
    robot_model.init_pose()
    ri.angle_vector(robot_model.angle_vector(), time=2.0, time_scale=1.0)
    ri.wait_interpolation()
    print("reset")

    joint: Joint = robot_model.r_wrist_roll_joint
    joint.joint_angle(math.pi * 4.0)
    ri.angle_vector(robot_model.angle_vector(), time=2.0, time_scale=1.0)
    ri.wait_interpolation()
    print("rotate")
