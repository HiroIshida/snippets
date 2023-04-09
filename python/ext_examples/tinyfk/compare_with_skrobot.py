import time
from tinyfk import BaseType, RobotModel
import numpy as np
import tinyfk
from skrobot.model import RobotModel as SkRobotModel
from skrobot.coordinates import Coordinates, rpy_matrix
from skrobot.models.pr2 import PR2

urdf_model_path = tinyfk.pr2_urdfpath()
kin_solver = tinyfk.RobotModel(urdf_model_path)
rarm_joint_names = ["r_shoulder_pan_joint", "r_shoulder_lift_joint", "r_upper_arm_roll_joint", "r_elbow_flex_joint", "r_forearm_roll_joint", "r_wrist_flex_joint", "r_wrist_roll_joint"]

rarm_joint_ids = kin_solver.get_joint_ids(rarm_joint_names)
end_link_id = kin_solver.get_link_ids(["r_gripper_tool_frame"])[0]

av = [-8.20535112e-01,  7.33814054e-01, -1.22169991e+00, -2.13501284e+00, -3.90940436e+00, -7.61528549e-01, -1.93254402e+00, -2.00827046e-01, -2.96483047e-01,  5.13190400e-01, -2.98544162e-03,  5.66229764e-01, -6.68729893e-02]


val, _ = kin_solver.solve_forward_kinematics([av], [end_link_id], rarm_joint_ids, True, BaseType.FLOATING)

pr2 = PR2(use_tight_joint_limit=False)
joint_angles, base_pose = av[:-6], av[-6:]
xyz, rpy = base_pose[:3], np.flip(base_pose[-3:])
co = Coordinates(pos = xyz, rot=rpy_matrix(*rpy))
pr2.newcoords(co)

for jn, an in zip(rarm_joint_names, av):
    pr2.__dict__[jn].joint_angle(an)
pos = pr2.__dict__["r_gripper_tool_frame"].worldpos()

np.testing.assert_almost_equal(val[0][:3], pos)
