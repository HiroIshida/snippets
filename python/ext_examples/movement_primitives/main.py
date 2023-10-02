import matplotlib.pyplot as plt
import numpy as np
from movement_primitives.dmp import DMP  # 0.5.0

start = np.zeros(2)
goal = np.array([1.0, 0.0])
n_split = 100
path = np.linspace(start, goal, n_split)
times = np.linspace(0, 1, n_split)

dmp = DMP(2, execution_time=1.0, n_weights_per_dim=10, dt=0.05)
dmp.imitate(times, path)

def create_traj():
    dmp.reset()
    x = np.random.randn(2) * 0.1
    dmp.configure(start_y=x, goal_y=goal)
    T, Y = dmp.open_loop()
    return Y

traj_list = [create_traj() for _ in range(100)]

fig, ax = plt.subplots()
for traj in traj_list:
    ax.plot(traj[:, 0], traj[:, 1], color="blue", alpha=0.1)
plt.show()




