import pybullet as p
import pybullet_data 
import numpy.random as rn
import numpy as np
import time

CLIENT = p.connect(p.GUI)#or pybullet.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

obj = p.loadURDF("./blue_can.urdf")
planeId = p.loadURDF("plane.urdf")

start = np.array([0.0, 0.0, 1.0])
direction = np.array([0.0, 0.0, -1.0])
result = p.rayTest(start, start + direction * 1.0)
