import rospy
from std_srvs.srv import *

rospy.init_node("client", anonymous=True)
rospy.wait_for_service("test")
client = rospy.ServiceProxy("test", Empty)
client()

