import time
import random
import os
import numpy as np
import pybullet
import pybullet_data
from predlearn.files import get_dataset_path
from predlearn.predicates import IsAbove, IsBelow
from predlearn.utils import pb_get_pose, pb_slide_point, temporary_slide, pb_set_pose
from predlearn.scripts.prepare_ycb import get_all_ycb_object_paths

client = pybullet.connect(pybullet.GUI)
pybullet.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS, 0)
pybullet.setGravity(0,0,-10)

planeId = pybullet.loadURDF("plane.urdf")

ycb_path = get_dataset_path('ycb') 
paths = get_all_ycb_object_paths()
random.shuffle(paths)

n_obj = 10 # 10 is the best
for path in paths[:n_obj]:
    random_shift = np.array([-0.1, -0.1, 0.0]) + np.random.rand(3) * np.array([0.2, 0.2, 0.3])
    pos = np.array([0, 0, 0.5]) + random_shift
    rpy = np.random.randn(3)
    quat = pybullet.getQuaternionFromEuler(rpy)
    obj_id = pybullet.loadURDF(path, basePosition=pos, baseOrientation=quat)

for i in range(50): # 30 steps for removing invalid overwraping state
    pybullet.stepSimulation()

for i in range(np.random.randint(low=0, high=200)): # 30 steps for removing invalid overwraping state
    pybullet.stepSimulation()

time.sleep(10)
