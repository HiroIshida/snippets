import skrobot
import numpy as np
from math import *
from skrobot.coordinates import Coordinates

robot_model = skrobot.models.urdf.RobotModelFromURDF(urdf_file=skrobot.data.pr2_urdfpath())
def initialize():
    robot_model.translation = np.zeros(3)
    robot_model.rotation = np.eye(3)
    robot_model.translate([1.0, 0.0, 0.0])

initialize()
robot_model.rotate(pi*0.5, 'z', 'world')
print(robot_model.worldpos())

initialize()
robot_model.rotate(pi*0.5, 'z', 'local')
print(robot_model.worldpos())
