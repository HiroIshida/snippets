import tqdm
import numpy as np
from cmaes import CMA
import datetime
import os
import pickle
from skrobot.models.pr2 import PR2
from skrobot.sdf.signed_distance_function import UnionSDF, link2sdf, trimesh2sdf
from skrobot.model.primitives import PointCloudLink
from skrobot.viewers import TrimeshSceneViewer
from sensor_msgs.msg import PointCloud2, JointState
import warnings

warnings.filterwarnings("ignore", message="violate")

with open("./pointcloud_20231124-025627.pkl", "rb") as f:
    pcloud, jstate = pickle.load(f)

pr2 = PR2(use_tight_joint_limit=False)

# reflect jstate
for joint_name, angle in zip(jstate.name, jstate.position):
    pr2.__dict__[joint_name].joint_angle(angle)

sdf_union = UnionSDF.from_robot_model(pr2)
pcloud_near = pcloud[sdf_union(pcloud) < 0.1]
assert len(pcloud_near) > 0

print(pr2.larm.joint_names)
pr2.l_shoulder_pan_joint.joint_angle(+0.03, relative=True)
pr2.l_shoulder_lift_joint.joint_angle(-0.04, relative=True)
pr2.l_upper_arm_roll_joint.joint_angle(-0.04, relative=True)
pr2.l_elbow_flex_joint.joint_angle(-0.13, relative=True)
pr2.l_forearm_roll_joint.joint_angle(0.0, relative=True)
pr2.l_wrist_flex_joint.joint_angle(0.1, relative=True)
pr2.l_wrist_roll_joint.joint_angle(0.0, relative=True)

plink = PointCloudLink(pcloud_near)
v = TrimeshSceneViewer()
v.add(plink)
v.add(pr2)
v.show()
import time; time.sleep(1000)
