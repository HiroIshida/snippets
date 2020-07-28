#!/usr/bin/env python
import rospy
from posedetection_msgs.msg import ObjectDetection

#topic_name = "/kinect_head/rgb/ObjectDetection"
topic_name = "/kinect_head/depth_registered/ObjectDetection"
topic_type = ObjectDetection

def callback(msg):
    rospy.loginfo("subscribing objectDetection")
    print("hoge")
    pass

rospy.init_node('dummy_subscriber')
rospy.Subscriber(topic_name, topic_type, callback)
rospy.spin()
