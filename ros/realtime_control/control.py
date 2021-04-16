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

        N = 6
        time_list = [0.2] * N
        interval = (self.goal_base_pose - self.base_pose)/N
        wpoints = [interval * (i+1) for i in range(N)]

        ret = self.ri.move_trajectory_sequence(wpoints, time_list, stop=False, 
                start_time=None, send_action=True, wait=False)
if __name__=='__main__':
    pc = PredictiveControler()
    rospy.spin()
