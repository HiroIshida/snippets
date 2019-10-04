
#!/usr/bin/python

import rospy
from geometry_msgs.msg import Wrench
import numpy as np

rospy.init_node('cartesian_wrench_test')

pub = rospy.Publisher('/arm_controller/cartesian_wrench/command', Wrench, queue_size=1)
msg = Wrench()
i = 0
while not rospy.is_shutdown():
    msg.force.x = 50 * np.sin(np.deg2rad(i * 30.0))
    msg.force.y = 50 * np.cos(np.deg2rad(i * 30.0))
    pub.publish(msg)
    rospy.sleep(0.1)
    i += 1
    i %= 360

