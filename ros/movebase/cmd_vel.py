import rospy
import time
from geometry_msgs.msg import *
rospy.init_node('talker', anonymous=True)
name_command = "/base_controller/command"

pub = rospy.Publisher(name_command, Twist, queue_size=1)
time.sleep(3)
rate = rospy.Rate(10)
while(True):
    msg = Twist()
    msg.linear.x = 0.5
    pub.publish(msg)
    rate.sleep()

