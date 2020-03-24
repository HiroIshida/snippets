
import rospy 
from std_srvs.srv import Empty
rospy.init_node("tmp", anonymous = True)

rospy.wait_for_service('freeze')
srvproxy = rospy.ServiceProxy('freeze', Empty)
srvproxy() # no arg cause Empty needs no arg
