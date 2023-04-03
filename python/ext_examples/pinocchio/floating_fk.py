import numpy as np
from robot_descriptions.jaxon_description import URDF_PATH
from robot_descriptions.loaders.pinocchio import load_robot_description
from skrobot.models.urdf import RobotModelFromURDF
from skrobot.model.primitives import Axis
from skrobot.coordinates import Coordinates
from skrobot.viewers import TrimeshSceneViewer
import tinyfk
import skrobot
from skrobot.model import Link
print(URDF_PATH)

model = load_robot_description("jaxon_description")
frame_table = {f.name: i for i, f in enumerate(model.model.frames)}
place = model.framePlacement(np.zeros(model.nq), frame_table["RARM_LINK7"])

co = Coordinates(pos=place.translation) 
axis = Axis.from_coords(co)

skmodel = RobotModelFromURDF(urdf_file=URDF_PATH)
viewer = TrimeshSceneViewer()
viewer.add(skmodel)
viewer.add(axis)
viewer.show()
