import tqdm
from scipy.linalg import block_diag, sqrtm
import numpy as np
import copy
import matplotlib.pyplot as plt
import numpy as np
from movement_primitives.dmp import DMP


class DMPMetric:
    def __init__(self, dmp_base: DMP):
        self.dmp_base = dmp_base

    @property
    def dim(self):
        return self.dmp_base.n_weights

    def __call__(self, param1, param2):
        dmp1 = copy.deepcopy(self.dmp_base)
        dmp1.forcing_term.weights_ += param1
        dmp2 = copy.deepcopy(self.dmp_base)
        dmp2.forcing_term.weights_ += param2
        dmp1.reset()
        dmp2.reset()
        _, traj1 = dmp1.open_loop()
        _, traj2 = dmp2.open_loop()
        dist = np.sum((traj1 - traj2) ** 2)
        return dist

def solve_lr_psd(X, y, n):
    N = X.shape[0]
    M = np.zeros((n, n))
    b = np.zeros(N)
    Xsym = []
    for i in range(N):
        xxT = np.outer(X[i], X[i])
        Xsym.append(xxT.flatten())
    Xmat = np.vstack(Xsym)
    a_hat, _, _, _ = np.linalg.lstsq(Xmat, y, rcond=None)
    A_hat = a_hat.reshape(n, n)
    A_hat = 0.5 * (A_hat + A_hat.T)
    w, v = np.linalg.eigh(A_hat)
    w[w < 0] = 0
    return v @ np.diag(w) @ v.T

start = np.zeros(1)
goal = np.ones(1)
n_split = 100  # this must be large
traj_original = np.linspace(start, goal, n_split)
times = np.linspace(0, 1, n_split)
dmp = DMP(1, execution_time=1.0, n_weights_per_dim=8, dt=0.01)
dmp.imitate(times, traj_original)
dmp.configure(start_y = start)

metric = DMPMetric(dmp)
n_sample = 1000
x1 = np.zeros(metric.dim)
X = np.random.randn(n_sample, metric.dim) * 10
Y = np.array([metric(x1, x) for x in tqdm.tqdm(X)])
A = solve_lr_psd(X, Y, metric.dim)
A_sqrt = sqrtm(A)
A_msqrt = np.linalg.inv(sqrtm(A))

Y_list = []
for _ in range(50):
    x = np.random.randn(metric.dim) * 0.3
    y = A_msqrt @ x
    tmp = copy.deepcopy(metric.dmp_base)
    tmp.forcing_term.weights_ += y
    _, Y = tmp.open_loop()
    Y_list.append(Y)

plt.figure()
for Y in Y_list:
    plt.plot(Y)
plt.show()
