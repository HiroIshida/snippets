import pinocchio as pin
from pathlib import Path
import copy
from tempfile import TemporaryDirectory
import numpy as np 
import xml.etree.ElementTree as ET
from robot_descriptions.jaxon_description import URDF_PATH
from robot_descriptions.loaders.pinocchio import load_robot_description
from skrobot.models.pr2 import PR2
from skrobot.models.urdf import RobotModelFromURDF
from skrobot.model.primitives import Axis
from skrobot.coordinates import Coordinates
from skrobot.viewers import TrimeshSceneViewer
import tinyfk
import skrobot
from skrobot.model import Link
import pinocchio as pin

np.random.seed(0)
# np.set_printoptions(precision=1)

urdf_model_path = tinyfk.pr2_urdfpath()

skmodel = RobotModelFromURDF(urdf_file=urdf_model_path)
joint_names = skmodel.joint_names

tree = ET.parse(urdf_model_path)
root = tree.getroot()
continuous_joints = root.findall(".//joint[@type='continuous']")
for joint in continuous_joints:
    joint.set("type", "revolute")
    limit = joint.find("limit")
    if limit is not None:
            limit.set("lower", "-10.0")
            limit.set("upper", "10.0")
            limit.set("effort", "10.0")
            limit.set("velocity", "10.0")
    else:
        new_limit = ET.Element("limit", {"lower": "-10.0", "upper": "10.0", "effort": "10.0", "velocity": "10.0"})
        joint.append(new_limit)

# remove 
with TemporaryDirectory() as td:
    td_path = Path(td)
    temp_urdf_path = td_path / "temp.urdf"
    tree.write(str(temp_urdf_path))
    tree.write("/tmp/pr2.urdf")

    model = pin.RobotWrapper.BuildFromURDF(
        filename=str(temp_urdf_path),
        package_dirs=None,
        root_joint=None,
    )

pin_frame_table = {f.name: i for i, f in enumerate(model.model.frames)}
pin_joint_table = {name: i for i, name in enumerate(model.model.names)}
del pin_joint_table["universe"]

arotttach_joint_name = "r_wrist_flex_joint"
attach_joint_id = pin_joint_table[attach_joint_name]
placement = pin.SE3.Identity()
new_frame = pin.Frame("new_link", attach_joint_id, 0, placement, pin.FrameType.OP_FRAME)
model.model.addFrame(new_frame, append_inertia=False)

