import tqdm
from pathlib import Path
import pybullet as pb
import time
import pybullet_data
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation


ignore_name_pairs = {
    ("shoulder_lift_link", "wrist_roll_link"),
    ("elbow_flex_link", "upperarm_roll_link"),
    ("base_link", "bellows_link2"),
    ("forearm_roll_link", "l_wheel_link"),
    ("estop_link", "forearm_roll_link"),
    ("bellows_link2", "shoulder_pan_link"),
    ("head_tilt_link", "shoulder_lift_link"),
    ("shoulder_lift_link", "shoulder_pan_link"),
    ("bellows_link2", "upperarm_roll_link"),
    ("forearm_roll_link", "l_gripper_finger_link"),
    ("base_link", "torso_fixed_link"),
    ("base_link", "estop_link"),
    ("head_tilt_link", "r_wheel_link"),
    ("estop_link", "l_wheel_link"),
    ("head_pan_link", "laser_link"),
    ("estop_link", "head_pan_link"),
    ("r_wheel_link", "torso_fixed_link"),
    ("l_wheel_link", "torso_fixed_link"),
    ("shoulder_lift_link", "upperarm_roll_link"),
    ("laser_link", "torso_fixed_link"),
    ("bellows_link2", "torso_lift_link"),
    ("torso_fixed_link", "torso_lift_link"),
    ("estop_link", "torso_fixed_link"),
    ("l_gripper_finger_link", "upperarm_roll_link"),
    ("head_pan_link", "shoulder_pan_link"),
    ("elbow_flex_link", "r_wheel_link"),
    ("forearm_roll_link", "wrist_roll_link"),
    ("elbow_flex_link", "r_gripper_finger_link"),
    ("base_link", "l_wheel_link"),
    ("gripper_link", "wrist_flex_link"),
    ("base_link", "head_pan_link"),
    ("base_link", "laser_link"),
    ("bellows_link2", "shoulder_lift_link"),
    ("l_wheel_link", "wrist_flex_link"),
    ("elbow_flex_link", "estop_link"),
    ("r_wheel_link", "wrist_roll_link"),
    ("gripper_link", "wrist_roll_link"),
    ("l_wheel_link", "laser_link"),
    ("l_wheel_link", "wrist_roll_link"),
    ("head_tilt_link", "l_wheel_link"),
    ("bellows_link2", "r_wheel_link"),
    ("l_wheel_link", "torso_lift_link"),
    ("elbow_flex_link", "forearm_roll_link"),
    ("forearm_roll_link", "shoulder_pan_link"),
    ("bellows_link2", "head_tilt_link"),
    ("forearm_roll_link", "shoulder_lift_link"),
    ("estop_link", "laser_link"),
    ("estop_link", "wrist_roll_link"),
    ("shoulder_pan_link", "torso_fixed_link"),
    ("head_pan_link", "torso_lift_link"),
    ("estop_link", "torso_lift_link"),
    ("gripper_link", "l_gripper_finger_link"),
    ("head_tilt_link", "torso_fixed_link"),
    ("elbow_flex_link", "gripper_link"),
    ("forearm_roll_link", "upperarm_roll_link"),
    ("elbow_flex_link", "wrist_flex_link"),
    ("l_wheel_link", "shoulder_pan_link"),
    ("base_link", "upperarm_roll_link"),
    ("elbow_flex_link", "l_wheel_link"),
    ("l_gripper_finger_link", "r_gripper_finger_link"),
    ("estop_link", "shoulder_pan_link"),
    ("elbow_flex_link", "wrist_roll_link"),
    ("head_pan_link", "shoulder_lift_link"),
    ("r_wheel_link", "upperarm_roll_link"),
    ("gripper_link", "upperarm_roll_link"),
    ("estop_link", "shoulder_lift_link"),
    ("l_wheel_link", "upperarm_roll_link"),
    ("shoulder_pan_link", "wrist_flex_link"),
    ("laser_link", "upperarm_roll_link"),
    ("base_link", "torso_lift_link"),
    ("estop_link", "upperarm_roll_link"),
    ("head_pan_link", "r_wheel_link"),
    ("elbow_flex_link", "l_gripper_finger_link"),
    ("r_wheel_link", "torso_lift_link"),
    ("head_pan_link", "head_tilt_link"),
    ("estop_link", "head_tilt_link"),
    ("head_tilt_link", "laser_link"),
    ("laser_link", "torso_lift_link"),
    ("base_link", "shoulder_pan_link"),
    ("base_link", "shoulder_lift_link"),
    ("l_gripper_finger_link", "wrist_flex_link"),
    ("bellows_link2", "torso_fixed_link"),
    ("r_gripper_finger_link", "wrist_flex_link"),
    ("bellows_link2", "estop_link"),
    ("r_wheel_link", "shoulder_pan_link"),
    ("wrist_flex_link", "wrist_roll_link"),
    ("r_wheel_link", "shoulder_lift_link"),
    ("l_gripper_finger_link", "l_wheel_link"),
    ("l_wheel_link", "shoulder_lift_link"),
    ("laser_link", "shoulder_pan_link"),
    ("forearm_roll_link", "r_gripper_finger_link"),
    ("head_tilt_link", "shoulder_pan_link"),
    ("l_gripper_finger_link", "wrist_roll_link"),
    ("laser_link", "shoulder_lift_link"),
    ("r_gripper_finger_link", "wrist_roll_link"),
    ("base_link", "r_wheel_link"),
    ("shoulder_lift_link", "torso_fixed_link"),
    ("base_link", "head_tilt_link"),
    ("l_wheel_link", "r_wheel_link"),
    ("upperarm_roll_link", "wrist_flex_link"),
    ("gripper_link", "r_gripper_finger_link"),
    ("laser_link", "r_wheel_link"),
    ("l_wheel_link", "r_gripper_finger_link"),
    ("head_pan_link", "l_wheel_link"),
    ("laser_link", "r_gripper_finger_link"),
    ("estop_link", "r_wheel_link"),
    ("upperarm_roll_link", "wrist_roll_link"),
    ("bellows_link2", "l_wheel_link"),
    ("shoulder_pan_link", "torso_lift_link"),
    ("bellows_link2", "head_pan_link"),
    ("estop_link", "r_gripper_finger_link"),
    ("bellows_link2", "laser_link"),
    ("elbow_flex_link", "shoulder_pan_link"),
    ("elbow_flex_link", "shoulder_lift_link"),
    ("shoulder_lift_link", "wrist_flex_link"),
    ("head_tilt_link", "torso_lift_link"),
    ("forearm_roll_link", "gripper_link"),
    ("head_pan_link", "torso_fixed_link"),
    ("forearm_roll_link", "wrist_flex_link"),
    ("r_gripper_finger_link", "upperarm_roll_link"),
}

physicsClient = pb.connect(pb.DIRECT)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())

# load robot from ~/.skrobot/fetch_description/fetch.urdf
robot = pb.loadURDF(str(Path("~/.skrobot/fetch_description/fetch.urdf").expanduser()))

def construct_tables():
    # construct tables
    # table (joint_name -> id) (link_name -> id)
    link_table = {pb.getBodyInfo(robot)[0].decode('UTF-8'):-1}
    joint_table = {}
    heck = lambda path: "_".join(path.split("/"))
    for _id in range(pb.getNumJoints(robot)):
        joint_info = pb.getJointInfo(robot, _id)
        joint_id = joint_info[0]

        joint_name = joint_info[1].decode('UTF-8')
        joint_table[joint_name] = joint_id
        name_ = joint_info[12].decode('UTF-8')
        name = heck(name_)
        link_table[name] = _id
    return link_table, joint_table

link_name_id_table, joint_name_id_table = construct_tables()
joint_id_name_table = {v: k for k, v in joint_name_id_table.items()}
link_id_name_table = {v: k for k, v in link_name_id_table.items()}


arm_links = ["shoulder_pan_link",
             "shoulder_lift_link",
             "upperarm_roll_link",
             "elbow_flex_link",
             "forearm_roll_link",
             "wrist_flex_link",
             "wrist_roll_link",
             "gripper_link",
             "r_gripper_finger_link",
             "l_gripper_finger_link"]
arm_link_ids = [link_name_id_table[ln] for ln in arm_links]
other_link_ids = set(link_name_id_table.values()) - set(arm_link_ids)
tmp = [(i, j) for i in arm_link_ids for j in other_link_ids]
id_pairs = set([(i, j) if i < j else (j, i) for i, j in tmp])

tmp = set([(link_name_id_table[name1], link_name_id_table[name2]) for name1, name2 in ignore_name_pairs])
ignore_id_pairs = set([(i, j) if i < j else (j, i) for i, j in tmp])
id_pairs = id_pairs - ignore_id_pairs

def minimum_distance(robotId):
    distance_th = 0.1
    num_joints = pb.getNumJoints(robotId)
    min_dist = distance_th
    for i, j in id_pairs:
        cps = pb.getClosestPoints(bodyA=robotId, bodyB=robotId, distance=distance_th, linkIndexA=i, linkIndexB=j)
        if len(cps) > 0:
            dist = min(cp[8] for cp in cps)
            if dist < min_dist:
                min_dist = dist
    return min_dist

ts = time.time()
for _ in range(1000):
    a = minimum_distance(robot)
print((time.time() - ts) / 1000)
