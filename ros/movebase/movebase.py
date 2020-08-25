import skrobot
import numpy as np
import time 
robot_model = skrobot.models.urdf.RobotModelFromURDF(urdf_file=skrobot.data.pr2_urdfpath())
ri = skrobot.interfaces.ros.PR2ROSRobotInterface(robot_model)
vel = np.array([0.5, 0, -0.2])
trajectory_points = np.array([vel * (i + 1) for i in range(4)])
time_list = [1.0]*4
time.sleep(2)
ri.move_trajectory_sequence(trajectory_points, time_list, stop=True, 
        start_time=None, send_action=True)
