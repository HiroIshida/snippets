import matplotlib.pyplot  as plt
import numpy.random as rn
import numpy as np

class Visualizer:
    def __init__(self, f):
        N = 200
        xlin = np.linspace(-6.0, 6.0, N)
        ylin = np.linspace(-6.0, 6.0, N)
        X, Y = np.meshgrid(xlin, ylin)

        self.N = N
        self.X = X
        self.Y = Y
        self.pts = np.array(list(zip(X.flatten(), Y.flatten())))
        self.f = f

    def __call__(self, x_seq=None, X=None):
        fs = self.f(self.pts)
        Z = fs.reshape((self.N, self.N))
        fig, ax = plt.subplots()
        ax.contourf(self.X, self.Y, Z, levels=[0, 10, 30, 50, 70, 90, 200, 300, 400, 600])

        if X is not None:
            ax.scatter(X[:, 0], X[:, 1], 'ro')

        if x_seq is not None:
            for i in range(len(x_seq) - 1):
                x0 = x_seq[i]
                x1 = x_seq[i+1]
                ax.plot([x0[0], x1[0]], [x0[1], x1[1]], 'ro')

        plt.show()

class CMA: 
    def __init__(self, fun):
        self.fun = fun
        self.C = np.eye(2) * 0.01
        self.mean_old = None
        self.mean = np.zeros(2)
        self.N_sample = 10

        # for visualization
        self.X = None

    def update(self):
        X = rn.multivariate_normal(self.mean, self.C, self.N_sample)
        self.X = X
        idx_min = np.argmin(self.fun(X))

        x_next = X[idx_min]

        p = x_next - self.mean
        C_rank1 = np.outer(p, p)

        self.C = self.C * 0.95 + C_rank1 * 0.05

        self.mean_old = self.mean
        self.mean = x_next


def himmelblau(X):
    return (X[:, 0]**2 + X[:, 1] - 11)**2 + (X[:, 0] + X[:, 1]**2 - 7)**2

optimizer = CMA(himmelblau)
vis = Visualizer(himmelblau)

for i in range(1000):
    optimizer.update()
    vis(optimizer.X)
