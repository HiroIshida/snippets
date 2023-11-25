import copy
import numpy as np
import matplotlib.pyplot as plt
from movement_primitives.dmp import DMP as _DMP


np.random.seed(1)

class DMP(_DMP):
    def set_param(self, param: np.ndarray) -> None:
        xytheta_scaling = np.array([0.01, 0.01, np.deg2rad(20.0)])
        goal_position_scaling = xytheta_scaling * 1.0

        xytheta_scaling = np.array([0.03, 0.03, np.deg2rad(20.0)])
        force_scalineg = xytheta_scaling * 300
        n_dim = 3
        n_goal_dim = 3
        W = param[:-n_goal_dim].reshape(n_dim, -1)
        self.forcing_term.weights_[:, :] += W[:, :] * force_scalineg[:, None]
        goal_param = param[-n_goal_dim:] * goal_position_scaling
        print(goal_param)
        self.goal_y += goal_param


n_split = 30
start = np.array([-0.06, -0.045, 0.0])
goal = np.array([-0.0, -0.045, 0.0])
diff_step = (goal - start) / (n_split - 1)
traj_default = np.array([start + diff_step * i for i in range(n_split)])
n_weights_per_dim = 6


fig, ax = plt.subplots(figsize=(8, 6))
for _ in range(6):
    dmp = DMP(3, execution_time=1.0, n_weights_per_dim=n_weights_per_dim, dt=0.05)
    dmp.imitate(np.linspace(0, 1, n_split), traj_default.copy())
    dmp.configure(start_y=traj_default[0])
    param = np.random.randn(3 * n_weights_per_dim + 3)
    dmp.reset()
    dmp.set_param(param)
    _, planer_traj = dmp.open_loop()

    x = planer_traj[:, 0]
    y = planer_traj[:, 1]
    yaw = planer_traj[:, 2]
    dx = np.cos(yaw) * 0.006
    dy = np.sin(yaw) * 0.006

    for i in range(len(x)):
        ax.plot([x[i], x[i] + dx[i]], [y[i], y[i] + dy[i]], color="blue")

    ax.plot(traj_default[:, 0], traj_default[:, 1], label="default")
    ax.plot(planer_traj[:, 0], planer_traj[:, 1], label="planer", color="red")
    ax.set_aspect("equal", "datalim")
plt.show()
