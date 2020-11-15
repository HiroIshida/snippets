import skrobot
import numpy as np
from skrobot.coordinates.math import rpy_matrix
from skrobot.coordinates import Coordinates

robot_model = skrobot.models.urdf.RobotModelFromURDF(urdf_file=skrobot.data.pr2_urdfpath())
rarm_end_coords = skrobot.coordinates.CascadedCoords(
        parent=robot_model.r_gripper_tool_frame, 
        name='rarm_end_coords')

print(rarm_end_coords.worldpos())
co = Coordinates(pos = [1.0, 1.0, 1.0], rot=rpy_matrix(0.3, 0.3, 0.3))
robot_model.newcoords(co)
print(rarm_end_coords.worldpos())
