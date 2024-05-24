import pinocchio as pin
import numpy as np
model = pin.buildModelFromUrdf("model.urdf", pin.JointModelFreeFlyer())
data = model.createData()
frame_id = model.getFrameId("link1")

def dq_dt(q: np.ndarray, omega: np.ndarray):
    qw, qx, qy, qz = q
    wx, wy, wz = omega
    dqdt = 0.5 * np.array([
        -qx * wx - qy * wy - qz * wz,
         qw * wx + qy * wz - qz * wy,
         qw * wy - qx * wz + qz * wx,
         qw * wz + qx * wy - qy * wx
    ])
    return dqdt

dt = 0.01
v = np.array([0.0, 0.0, 0.0, 0.3, 0.2, 0.1])


# quaternion propagation using Pinocchio
quat_list = []
q = np.array([0, 0, 0, 1, 0, 0, 0])
for i in range(3000):
    q = pin.integrate(model, q, v * dt)
    quat_list.append(q[3:])

# quaternion propagation using numerical integration using analytical formula
quat_list2 = []
quat = np.array([1., 0, 0, 0])
for i in range(3000):
    dquat = dq_dt(quat, v[3:]) * dt
    quat = quat + dquat
    quat_list2.append(quat)

import matplotlib.pyplot as plt
Q1 = np.array(quat_list)
Q2 = np.array(quat_list2)
plt.plot(Q1[:, 2], "r.")
plt.plot(Q2[:, 2], "b--")
plt.show()
