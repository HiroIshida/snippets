import tf
import rospy
import numpy as np
import skrobot

if __name__=='__main__':
    robot_model = skrobot.models.PR2()
    robot_model.reset_manip_pose()
    ri = skrobot.interfaces.ros.PR2ROSRobotInterface(robot_model)

    step = np.array([0.3, 0.0, 0.0])
    wpoints = np.array([np.zeros(3) + step * i for i in range(10)])
    time_list = [3.0] * 10
    ret = ri.move_trajectory_sequence(wpoints, time_list, stop=False, 
            start_time=None, send_action=True, wait=False)

