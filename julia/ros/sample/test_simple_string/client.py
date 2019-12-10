import rospy
from sample.srv import *

rospy.init_node("client", anonymous=True)
rospy.wait_for_service("test")
client = rospy.ServiceProxy("test", SimpleString)
req = SimpleStringRequest("hello")
client(req)

