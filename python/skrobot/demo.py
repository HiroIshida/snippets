#!/usr/bin/env python
import time
import numpy as np
import pybullet
import skrobot
import rospkg
rospack = rospkg.RosPack()
models_dir = rospack.get_path("eusurdf")

robot = skrobot.models.Fetch()
interface = skrobot.interfaces.PybulletRobotInterface(robot)

table_file = models_dir + "/models/room73b2-karimoku-table/model.urdf"
table = pybullet.loadURDF(table_file, basePosition = [1, 0.0, 0.0])

for _ in range(100):
    pybullet.stepSimulation()
time.sleep(3)


