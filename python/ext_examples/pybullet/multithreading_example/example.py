#!/usr/bin/env python3
import threading
import numpy as np
import time
from typing import Dict, List

import pybullet as pb
import pybullet_data

pb.connect(pb.GUI)  # or pybullet.DIRECT for non-graphical version
pb.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
pb.loadURDF("plane.urdf")
pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS, 0)
pb.setGravity(0, 0, -10)
robot_id = pb.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

joint_table_all = {}
for idx in range(pb.getNumJoints(robot_id)):
    joint_info = pb.getJointInfo(robot_id, idx)
    joint_id = joint_info[0]
    joint_name = joint_info[1].decode("UTF-8")
    joint_table_all[joint_name] = joint_id

home_position = {
    "panda_joint1": 0.0,
    "panda_joint2": 0.7,
    "panda_joint3": 0.0,
    "panda_joint4": -0.5,
    "panda_joint5": 0.0,
    "panda_joint6": 1.3,
    "panda_joint7": -0.8,
}

home_position2 = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.2,
    "panda_joint3": 0.0,
    "panda_joint4": -0.5,
    "panda_joint5": 0.0,
    "panda_joint6": 1.3,
    "panda_joint7": -0.8,
}

share_dict = {"thread_running": True}

class Simulator(threading.Thread):
    def __init__(self):
        super().__init__()

    def run(self):
        while share_dict["thread_running"]:
            time.sleep(0.005)
            pb.stepSimulation()
        print("quit")


t1 = Simulator()
t1.start()

for _ in range(10):
    for key, val in home_position.items():
        joint_id = joint_table_all[key]
        pb.setJointMotorControl2(bodyIndex=robot_id, jointIndex=joint_id, controlMode=pb.POSITION_CONTROL, targetPosition=val, targetVelocity=0.0,
            force=200, positionGain=0.3, velocityGain=1.0, maxVelocity=1.0)
    time.sleep(1)

    for key, val in home_position2.items():
        joint_id = joint_table_all[key]
        pb.setJointMotorControl2(bodyIndex=robot_id, jointIndex=joint_id, controlMode=pb.POSITION_CONTROL, targetPosition=val, targetVelocity=0.0,
            force=200, positionGain=0.3, velocityGain=1.0, maxVelocity=1.0)
    time.sleep(1)

time.sleep(10)
share_dict["thread_running"] = False
time.sleep(2)

