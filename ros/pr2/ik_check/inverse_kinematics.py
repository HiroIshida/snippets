import rospy
import time
from geometry_msgs.msg import _PoseStamped, PoseStamped
from rospy import Publisher
from skrobot.models.pr2 import PR2 
from skrobot.interfaces.ros import PR2ROSRobotInterface
from skrobot.coordinates import Coordinates
import numpy as np

if __name__ == '__main__':
    robot = PR2()
    robot.reset_manip_pose()
    ri = PR2ROSRobotInterface(robot)

    arm = "larm"
    if arm == "rarm":
        link_list = robot.rarm.link_list
        move_target = robot.rarm_end_coords
    else:
        link_list = robot.larm.link_list
        move_target = robot.larm_end_coords


    pub = Publisher('/target_pose', PoseStamped, queue_size=1, latch=True)
    for _ in range(100):
        x = np.random.uniform(0.3, 0.9)
        if arm == "rarm":
            y = np.random.uniform(-0.7, 0.3)
        else:
            y = np.random.uniform(-0.3, 0.7)
        z = np.random.uniform(0.3, 1.0)
        target_coords = Coordinates([x, y, z])

        ret = robot.inverse_kinematics(
            target_coords, link_list=link_list, move_target=move_target
        )
        if isinstance(ret, bool) and ret is False:
            print("inverse kinematics failed")
            continue
        ri.angle_vector(robot.angle_vector(), time_scale=3.0)
        ri.wait_interpolation()

        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "base_footprint"
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.w = 1.0
        pub.publish(pose)

        time.sleep(3.0)

