import time
from skrobot.models import PR2
from skrobot.interfaces.ros import PR2ROSRobotInterface
from pr2_controllers_msgs.msg import JointControllerState

robot_model = PR2()
robot_interface = PR2ROSRobotInterface(robot_model)
robot_model.angle_vector(robot_interface.angle_vector())

while True:
    time.sleep(0.5)
    rarm_state: JointControllerState = robot_interface.gripper_states['rarm']
    larm_state: JointControllerState = robot_interface.gripper_states['larm']
    l_gripper_pos = larm_state.process_value
    r_gripper_pos = rarm_state.process_value
    target = larm_state.process_value
    print("current: {}".format(r_gripper_pos))
    print("target: {}".format(target))

    pseudo_target = r_gripper_pos + (target - r_gripper_pos) * 1.0
    target_rarm_gripper_pos = rarm_state.process_value
    robot_interface.move_gripper('rarm', pseudo_target, effort=100, wait=False)
