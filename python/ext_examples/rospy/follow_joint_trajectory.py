import rospy
import actionlib
import control_msgs.msg

rospy.init_node('client')
client = actionlib.SimpleActionClient('l_arm_controller/follow_joint_trajectory', control_msgs.msg.FollowJointTrajectoryAction)
client.wait_for_server()
