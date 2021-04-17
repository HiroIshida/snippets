import tf
import rospy
import numpy as np
import skrobot

from nav_msgs.msg import Odometry

class PredictiveControler(object):
    def __init__(self):

        robot_model = skrobot.models.PR2()
        robot_model.reset_manip_pose()
        ri = skrobot.interfaces.ros.PR2ROSRobotInterface(robot_model)

        # must come after ri
        rospy.Subscriber("/base_pose_ground_truth", Odometry, self._callback)

        self.timer = rospy.Timer(rospy.Duration(0.1), self._controller)
        self.base_pose = None # np.array 2d
        self.goal_base_pose = np.array([2.0, 2.0, 1.0])
        self.robot_model = robot_model
        self.ri = ri
        self.max_vel = np.array([0.4, 0.4, 0.4])
        self.dt = 0.2
        self.steps_horizon = 10
       
    def _callback(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        q = (ori.x, ori.y, ori.z, ori.w)
        rpy = tf.transformations.euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        yaw = rpy[2]
        self.base_pose = np.array([pos.x, pos.y, yaw])

    def _controller(self, msg):
        print("controller is called")
        if self.base_pose is None:
            return

        diff = self.goal_base_pose - self.base_pose
        sec_required = diff/self.max_vel
        float_step_required = sec_required/self.dt
        int_step_required = np.floor(float_step_required)
        remain_step_required = float_step_required - int_step_required

        wpoints = np.zeros((self.steps_horizon, 3))

        for i in range(3):
            isr = int(int_step_required[i])
            const_interval = self.max_vel[i] * self.dt

            reachable_in_horizon = (isr + 1 <= self.steps_horizon)
            if reachable_in_horizon:
                wpoints[:isr, i] = np.array([const_interval * (j + 1) for j in range(isr)])
                wpoints[isr, i] = diff[i]
                #from IPython import embed; embed()
            else:
                wpoints[:, i] = np.array([const_interval * (j + 1) for j in range(self.steps_horizon)])
        #import sys; sys.exit()
        print("=====LOG=======")
        print(diff)
        print(wpoints)

        time_list = [self.dt * i for i in range(self.steps_horizon)]
        ret = self.ri.move_trajectory_sequence(wpoints, time_list, stop=False, 
                start_time=None, send_action=True, wait=False)

if __name__=='__main__':
    pc = PredictiveControler()
    rospy.spin()
    """
    a = np.array([3., 3, 3])
    b = np.array([1., 2., 3])
    """
