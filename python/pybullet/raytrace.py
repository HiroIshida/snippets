# https://towardsdatascience.com/simulate-images-for-ml-in-pybullet-the-quick-easy-way-859035b2c9dd
import pybullet as p
import pybullet_data 
import numpy.random as rn
import numpy as np
import time
import rospkg
import matplotlib.pyplot as plt

def get_pose(body):
    return p.getBasePositionAndOrientation(body, physicsClientId=CLIENT)

def get_quat(body):
    return get_pose(body)[1] # [x,y,z,w]

def get_point(body):
    return get_pose(body)[0]

def set_pose(body, pose):
    (point, quat) = pose
    p.resetBasePositionAndOrientation(body, point, quat, physicsClientId=CLIENT)

def set_point(body, point):
    set_pose(body, (point, get_quat(body)))

rospack = rospkg.RosPack()

CLIENT = p.connect(p.GUI)#or pybullet.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

model_urdf = rospack.get_path("eusurdf")
oven = p.loadURDF(model_urdf+"/models/cup/model.urdf")
hoge = p.loadURDF("plane.urdf")

direction = np.array([1.0, 0.0, 0.0])

# umakuiku
set_point(oven, [0, 0.0, 0.0])
pos = np.array([-0.5 , 0.0  , 0.02])
result1 = p.rayTest(pos, pos + direction * 0.5)[0]

# umakuikanai girigiri
set_point(oven, [0, 0.0, 0.0])
pos = np.array([-0.05 , 0.0  , 0.02])
result2 = p.rayTest(pos, pos + direction * 0.5)[0]

col_pos1 = result1[3]
col_pos2 = result2[3]


