#!/usr/bin/env python

import numpy as np
import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
import tf
from geometry_msgs.msg import TransformStamped
from skrobot.coordinates.math import rpy2quaternion, wxyz2xyzw


class DummyInfoPublisher:

    def __init__(self):
        # remote indicates the images are uncompressed
        self.sub_info = rospy.Subscriber("/k4a/rgb/camera_info", CameraInfo, self.callback_info)
        self.sub_img = rospy.Subscriber("/remote/k4a/rgb/image_rect_color", Image, self.callback_image)
        self.pub_info = rospy.Publisher("/remote/k4a/rgb/camera_info", CameraInfo, queue_size=10)
        self.info = None

    def callback_info(self, msg):
        self.info = msg
        self.sub_info.unregister()  # one shot

    def callback_image(self, msg: Image):
        if self.info is None:
            return
        self.info.header = msg.header
        self.pub_info.publish(self.info)
        rospy.loginfo("published camera info")


if __name__ == "__main__":

    rospy.init_node("test_synchronization")
    pub = DummyInfoPublisher()

    br = tf.TransformBroadcaster()
    rate = rospy.Rate(10.0)

    while not rospy.is_shutdown():
        translation = (0.19, 0.03, -0.03)  # Example values
        # wxyz = rpy2quaternion((0.0, np.pi * 0.5, 0.0))
        wxyz = rpy2quaternion([0.0, 0.0, 0.0])
        print(wxyz)
        rotation = tuple(wxyz2xyzw(wxyz))
        # rotate 180 degrees around the y axis

        br.sendTransform(translation, rotation, rospy.Time.now(), "camera_base", "head_mount_link")
        rate.sleep()

    rospy.spin()
