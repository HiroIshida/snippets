import numpy as np
import time
from skrobot.models import PR2
from skrobot.interfaces.ros import PR2ROSRobotInterface
from pr2_controllers_msgs.msg import JointControllerState

robot_model = PR2()
robot_interface = PR2ROSRobotInterface(robot_model)
robot_model.angle_vector(robot_interface.angle_vector())

phase = np.linspace(0, 4 * np.pi, 100)
for p in phase:
    width = np.sin(p) * 0.04 + 0.04
    print(width)
    robot_interface.move_gripper('rarm', width, effort=25, wait=False)
    time.sleep(0.1)
