import numpy as np
from robot_descriptions.loaders.pinocchio import load_robot_description
import pinocchio as pin

robot: pin.RobotWrapper = load_robot_description("jaxon_description")
q = np.zeros(robot.nq)
v = np.zeros(robot.nv)  # -1 dimension as q because quaternion parameterization stuff
a = np.zeros(robot.nv)  # same dimension as v
pin.computeCentroidalMomentum(robot.model, robot.data, q, v)
print(robot.data.hg)  # momentum
print(robot.data.hg.vector)  # momentum
print(robot.data.com[0])  # center of mass
print(robot.data.vcom[0])  # velocity of center of mass

out = pin.computeCentroidalDynamicsDerivatives(robot.model, robot.data, q, v, a)
dh_dq = out[0]
print(dh_dq.shape)
