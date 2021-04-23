import time
import rospy
from geometry_msgs.msg import Twist

pub = rospy.Publisher("/base_controller/command", Twist, queue_size=1)
rospy.init_node('tmp', anonymous=True)

tw = Twist() 
tw.linear.x = 1.0
for i in range(30):
    print("publishing")
    pub.publish(tw)
    time.sleep(0.2)
