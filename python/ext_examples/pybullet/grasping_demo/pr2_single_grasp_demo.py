from math import *
import pybullet 
import pybullet as pb
import pybullet_data
import numpy as np
import time
import utils

from gripper import RGripper, LGripper


try:
    isInit
except:
    isInit = True
    CLIENT = pybullet.connect(pybullet.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
    plane = pybullet.loadURDF("plane.urdf")
    pb.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0, physicsClientId=CLIENT)
    pb.configureDebugVisualizer(pybullet.COV_ENABLE_TINY_RENDERER, 0, physicsClientId=CLIENT)
    pb.setGravity(0,0,-10.0)
    table = pb.loadURDF("table/table.urdf", physicsClientId=CLIENT)
    dish = pb.loadURDF("dish/dish.urdf", physicsClientId=CLIENT)
    rgripper = RGripper()

table_pos = np.array([0.0, 0.0, 0.0])
utils.set_point(table, table_pos)
utils.set_zrot(table, pi*0.5)

utils.set_point(dish, [0.0, 0.0, 0.63])
rgripper.set_basepose([0, 0.15, 0.86], [-1.54, 1.2, -1.57])
rgripper.set_gripper_width(0.5, force=True)

time.sleep(0.5)
rgripper.set_gripper_width(0.0)
rgripper.set_state([0.0, 0.0, -0.2])
rgripper.set_state([-0.2, 0.5, 0.0])
time.sleep(0.5)

for i in range(300):
    pb.stepSimulation(physicsClientId=CLIENT)
    time.sleep(0.005)

rgripper.set_state([0.0, -0.5, 0.0])
for i in range(1000):
    pb.stepSimulation(physicsClientId=CLIENT)
    time.sleep(0.005)
