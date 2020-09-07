# require https://github.com/CyberAgent/cmaes
import matplotlib.pyplot  as plt
import scipy.optimize as opt
import numpy as np

# also use recent one
#def rank1update(X):

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

    def __call__(self, x_seq=None):
        fs = self.f(self.pts[:, 0], self.pts[:, 1])
        Z = fs.reshape((self.N, self.N))
        fig, ax = plt.subplots()
        ax.contourf(self.X, self.Y, Z, levels=[0, 10, 30, 50, 70, 90, 200, 300, 400, 600])

        if x_seq is not None:
            for i in range(len(x_seq) - 1):
                x0 = x_seq[i]
                x1 = x_seq[i+1]
                ax.plot([x0[0], x1[0]], [x0[1], x1[1]], 'ro')

        plt.show()

"""
def rosenbrock(x1, x2):
    return 100 * (x2 - x1 ** 2) ** 2 + (x1 - 1) ** 2
"""

def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def quadratic(x, y):
    return (x - 5.0)**2 + (y - 5.0)**2

import numpy as np
from cmaes import CMA

fun = himmelblau
vis = Visualizer(fun)

if __name__ == "__main__":
    optimizer = CMA(mean=np.zeros(2), sigma=1.3)

    x_seq = []

    for generation in range(50):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            x_seq.append(optimizer._mean)
            if len(x_seq)%10==1:
                vis(x_seq)
            print(optimizer._C)
            value = fun(x[0], x[1])
            solutions.append((x, value))
            print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]})")
        optimizer.tell(solutions)
