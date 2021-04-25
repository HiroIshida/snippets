import time
import rospy
from geometry_msgs.msg import Twist
from pr2_mechanism_controllers.msg import BaseDirectCommand

pub = rospy.Publisher("base_controller/direct_command", BaseDirectCommand, queue_size=1)
rospy.init_node('tmp', anonymous=True)

d_cmd = BaseDirectCommand()
d_cmd.wheel_vels = [10.0 for _ in range(8)]
d_cmd.caster_vels = [0.0 for _ in range(4)]

while True:
    print("publishing")
    pub.publish(d_cmd)
    time.sleep(0.2)
