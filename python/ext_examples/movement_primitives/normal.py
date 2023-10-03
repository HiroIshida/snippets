import matplotlib.pyplot as plt
import numpy as np
from movement_primitives.dmp import DMP  # 0.5.0

start = np.zeros(2)
goal = np.array([1.0, 0.2])
n_split = 200  # this must be large
traj_original = np.linspace(start, goal, n_split)
times = np.linspace(0, 1, n_split)

dmp = DMP(2, execution_time=1.0, n_weights_per_dim=10, dt=0.1)
dmp.imitate(times, traj_original)

dmp.reset()
dmp.configure(start_y = start)
_, traj_reproduced = dmp.open_loop()

def create_traj():
    dmp.reset()
    x = np.random.randn(2) * 0.1
    dmp.configure(start_y=x)
    T, Y = dmp.open_loop()
    return Y

traj_list = [create_traj() for _ in range(100)]

fig, ax = plt.subplots()
for traj in traj_list:
    ax.plot(traj[:, 0], traj[:, 1], color="blue", alpha=0.1)
ax.plot(traj_reproduced[:, 0], traj_reproduced[:, 1], "ro-")
ax.plot(traj_original[:, 0], traj_original[:, 1], "g.-")
plt.show()
