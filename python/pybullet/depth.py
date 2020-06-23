# https://towardsdatascience.com/simulate-images-for-ml-in-pybullet-the-quick-easy-way-859035b2c9dd
import pybullet as p
import pybullet_data 
import numpy.random as rn
import numpy as np
import time
import rospkg
import matplotlib.pyplot as plt

rospack = rospkg.RosPack()

CLIENT = p.connect(p.DIRECT)#or pybullet.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

model_urdf = rospack.get_path("eusurdf")
oven = p.loadURDF(model_urdf+"/models/cup/model.urdf")
hoge = p.loadURDF("plane.urdf")

viewMatrix = p.computeViewMatrix(
    cameraEyePosition=[0.3, 0, 0.1],
    cameraTargetPosition=[0, 0, 0],
    cameraUpVector=[0, 1, 0])

projectionMatrix = p.computeProjectionMatrixFOV(
    fov=45.0,
    aspect=1.0,
    nearVal=0.1,
    farVal=3.1)

width, height, rgbImg, depthImg, segImg = p.getCameraImage(
    width=224, 
    height=224,
    viewMatrix=viewMatrix,
    projectionMatrix=projectionMatrix)

plt.imshow(depthImg)
