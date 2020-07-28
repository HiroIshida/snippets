#!/usr/bin/env python
import rospy
from posedetection_msgs.msg import ObjectDetection

prefix = "/head_camera_remote/"
topic_name = prefix + "rgb/ObjectDetection"
topic_type = ObjectDetection

def callback(msg):
    #rospy.loginfo("subscribing objectDetection")
    pass

rospy.init_node('dummy_subscriber')
rospy.Subscriber(topic_name, topic_type, callback)
rospy.spin()

