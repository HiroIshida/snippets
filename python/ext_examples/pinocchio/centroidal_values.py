import numpy as np
from robot_descriptions.loaders.pinocchio import load_robot_description
import pinocchio as pin

np.random.seed(0)

robot: pin.RobotWrapper = load_robot_description("jaxon_description")
q = pin.randomConfiguration(robot.model)
v = np.random.rand(robot.nv) 
a = np.zeros(robot.nv)  # same dimension as v
h = pin.computeCentroidalMomentum(robot.model, robot.data, q, v)
print(f"Centroidal momentum: {h.vector.shape}")
print(f"cached centroidal momentum: {robot.data.com[0].shape}")
print(f"cached centroidal momentum velocity: {robot.data.vcom[0].shape}")

Ag = pin.computeCentroidalMap(robot.model, robot.data, q)
print(f"centroidal momentum matrix: {Ag.shape}")

# following momentum dynamics equation
residual = h.vector - Ag.dot(v) 
print(h.vector)
assert np.allclose(residual, np.zeros(6))
print("test on momentum dynamics equation passed")

# dh_dq nuemrical-analytical comparison
def compute_dh_dq_numerical(q, v, a, robot):
    eps = 1e-6
    dh_dq = np.zeros((6, robot.nq))
    dh0 = pin.computeCentroidalMomentum(robot.model, robot.data, q, v)
    for i in range(robot.nq):
        q_plus = q.copy()
        q_plus[i] += eps
        dh_plus = pin.computeCentroidalMomentum(robot.model, robot.data, q_plus, v)
        dh_dq_i = (dh_plus.vector - dh0.vector) / eps
        dh_dq[:, i] = dh_dq_i
    return dh_dq

out = pin.computeCentroidalDynamicsDerivatives(robot.model, robot.data, q, v, a)
dh_dq = out[0]
dh_dq_num = compute_dh_dq_numerical(q, v, a, robot)
assert np.allclose(dh_dq, dh_dq_num, atol=1e-4)
print("test on dh_dq passed")
