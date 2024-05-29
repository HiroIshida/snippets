import time
import numpy as np
from robot_descriptions.loaders.pinocchio import load_robot_description
import pinocchio as pin

np.random.seed(0)
robot: pin.RobotWrapper = load_robot_description("jaxon_description", root_joint=pin.JointModelFreeFlyer())
# robot: pin.RobotWrapper = load_robot_description("jaxon_description")
q = pin.randomConfiguration(robot.model)
q[:3] = 0.0
v = np.random.rand(robot.nv) 
a = np.zeros(robot.nv)  # whatever
a = np.random.rand(robot.nv)
h = pin.computeCentroidalMomentum(robot.model, robot.data, q, v)
print(f"Centroidal momentum: {h.vector.shape}")
print(f"h: {h.vector}")
print(f"cached centroidal momentum: {robot.data.com[0].shape}")
print(f"cached centroidal momentum velocity: {robot.data.vcom[0].shape}")

Ag = pin.computeCentroidalMap(robot.model, robot.data, q)
print(f"centroidal momentum matrix: {Ag.shape}")

# following momentum dynamics equation
residual = h.vector - Ag.dot(v) 
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

    # quaternion kinematics
    q1, q2, q3, q4 = q[3:7]
    E = 0.5 * np.array([[q4, -q3, q2], [q3, q4, -q1], [-q2, q1, q4], [-q1, -q2, -q3]])  # 4 x 3
    T = np.zeros((robot.nq, robot.nv))
    T[:3, :3] = np.eye(3)
    T[3:7, 3:6] = E
    T[7:, 6:] = np.eye(robot.nv - 6)
    return dh_dq.dot(T)


def dh_dq(q: np.ndarray, v: np.ndarray, robot: pin.RobotWrapper):
    out = pin.computeCentroidalDynamicsDerivatives(robot.model, robot.data, q, v, a)
    dh_dq = out[0]
    return dh_dq

val = dh_dq(q, v, robot)
dh_dq_num = compute_dh_dq_numerical(q, v, a, robot)
diff = val - dh_dq_num 
diff_bool = np.abs(diff) > 1e-4
assert not np.any(diff_bool)
print("test on dh_dq passed")
