import skrobot
import time
import numpy as np

robot_model = skrobot.models.PR2()
robot_model.reset_manip_pose()
robot_model.head_tilt_joint.joint_angle(0.4)

viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(641, 480))
viewer.add(robot_model)

robot_model.fksolver = None
ri = skrobot.interfaces.ros.PR2ROSRobotInterface(robot_model)

time_seq = [20.0] * 2
base_traj = np.array([[0.0, 3.0, 0.0], [0.0, 6.0, 0.0]])
ri.move_trajectory_sequence(base_traj, time_seq, send_action=True)
