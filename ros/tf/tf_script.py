#!/usr/bin/env python
import rospy
from tf2_msgs.msg import TFMessage
import dill
import sys

rospy.init_node("tmp", anonymous=True)

def cb(msg):
    with open("tf.dill", "w") as f:
        print("saving")
        dill.dump(msg, f)
    sys.exit()
sub = rospy.Subscriber("/tf", TFMessage, cb, queue_size=10)
rospy.spin()
