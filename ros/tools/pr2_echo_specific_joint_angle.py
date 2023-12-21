#!/usr/bin/env python
import time
import rospy
from sensor_msgs.msg import JointState
import argparse

class Node:

    def __init__(self, joint_name: str):
        rospy.Subscriber("/joint_states", JointState, callback=self.callback)
        self.name_idx_map = None
        self.joint_name = joint_name

    def callback(self, msg):
        if self.name_idx_map is None:
            self.name_idx_map = {name: i for (i, name) in enumerate(msg.name)}
        rospy.loginfo(f"{self.joint_name}: {msg.position[self.name_idx_map[self.joint_name]]}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="torso_lift_joint")
    args = parser.parse_args()

    rospy.init_node("pr2_specific_joint_angle_echo")
    node = Node(args.name)
    rospy.spin()
