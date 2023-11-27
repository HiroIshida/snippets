#!/usr/bin/env python
import numpy as np
import rospy
import tf
from geometry_msgs.msg import PoseStamped
from skrobot.coordinates.math import rpy2quaternion, wxyz2xyzw, quaternion2rpy, xyzw2wxyz

def publish_pose(listener, publisher, target_frame, source_frame):
    try:
        (trans, rot) = listener.lookupTransform(target_frame, source_frame, rospy.Time(0))

        pose = PoseStamped()
        pose.header.frame_id = target_frame
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = trans[0]
        pose.pose.position.y = trans[1]
        pose.pose.position.z = trans[2]
        # we care only yaw and know that othere angles are 0
        ypr = quaternion2rpy(xyzw2wxyz(rot))[0]
        ypr[1] = 0.0
        # ypr[2] = np.pi
        rot = rpy2quaternion(ypr)
        pose.pose.orientation.x = rot[0]
        pose.pose.orientation.y = rot[1]
        pose.pose.orientation.z = rot[2]
        pose.pose.orientation.w = rot[3]

        publisher.publish(pose)
        rospy.loginfo("Published pose")
    
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        rospy.loginfo("TF Exception")

def main():
    rospy.init_node('pose_publisher_node')

    target_frame = "base_footprint"
    source_frame = "object"

    listener = tf.TransformListener()
    pose_pub = rospy.Publisher('april_pose', PoseStamped, queue_size=10)
    
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        publish_pose(listener, pose_pub, target_frame, source_frame)
        rate.sleep()

if __name__ == '__main__':
    main()
