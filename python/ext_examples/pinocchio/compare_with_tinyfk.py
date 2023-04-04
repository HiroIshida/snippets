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
  
for nq, name in zip(model.model.nqs, model.model.names):
    print("{}: {}".format(name, nq))

pin_frame_table = {f.name: i for i, f in enumerate(model.model.frames)}
pin_joint_table = {name: i for i, name in enumerate(model.model.names)}
del pin_joint_table["universe"]

# setup tinyfk solver
kin_solver = tinyfk.RobotModel(urdf_model_path)

attach_joint_name = "r_wrist_flex_joint"
attach_frame_name = "r_wrist_flex_link"

## add link to tinyfk
parent_link_id = kin_solver.get_link_ids(["r_wrist_flex_link"])[0]
kin_solver.add_new_link("new_link", parent_link_id, [0.1, 0.1, 0.1], [0, 0, 0]);

# add link (new_link) to pinnochio model
attach_joint_id = pin_joint_table[attach_joint_name]
attach_link_id = pin_frame_table[attach_frame_name]

## first, get relative transfrom from joint to the frame
joint_placement = model.placement(np.zeros(model.nq), attach_joint_id)
world_placement = model.framePlacement(np.zeros(model.nq), attach_link_id)
frame_placement = world_placement.inverse() * joint_placement

placement = pin.SE3.Identity()
placement.translation = np.array([0.1, 0.1, 0.1])
new_frame = pin.Frame("new_link", attach_joint_id, 0, placement, pin.FrameType.OP_FRAME)
model.model.addFrame(new_frame, append_inertia=False)

pin_frame_table = {f.name: i for i, f in enumerate(model.model.frames)}  # update table
assert "new_link" in pin_frame_table

pr2 = PR2()
pr2.reset_manip_pose()
angles = {jn: pr2.__dict__[jn].joint_angle() for jn in joint_names}

for link_name in ["new_link", "r_wrist_flex_link"]:
    # compute tinyfk fk
    sk_av = np.array(list(angles.values()))
    joint_ids = kin_solver.get_joint_ids(joint_names)
    elink_ids = kin_solver.get_link_ids([link_name])
    P_tinyfk, _ = kin_solver.solve_forward_kinematics(np.expand_dims(sk_av, axis=0), elink_ids, joint_ids)

    # check if tinyfk fk and pinocchio fk are equal
    pin_av = np.array([angles[name] for name, idx in pin_joint_table.items()])
    model.forwardKinematics(pin_av)
    P0 = model.framePlacement(pin_av, pin_frame_table[link_name])
    np.testing.assert_almost_equal(P_tinyfk[0], P0.translation, decimal=5)
     
    # check if numerical and analytical jacobian equal
    jac_numel = np.zeros((3, len(pin_av)))
    P0 = model.framePlacement(pin_av, pin_frame_table[link_name])
    model.computeJointJacobians(pin_av)
    
    jac_pin_anal = model.getFrameJacobian(pin_frame_table[link_name], rf_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3]
    eps = 1e-6
    for i in range(len(pin_av)):
        pin_av_copied = copy.deepcopy(pin_av)
        pin_av_copied[i] += eps
        P1 = model.framePlacement(pin_av_copied, pin_frame_table[link_name])
        jac_numel[:, i] = (P1.translation - P0.translation) / eps
    
    np.testing.assert_almost_equal(jac_numel, jac_pin_anal, decimal=5)

print("test passed")
