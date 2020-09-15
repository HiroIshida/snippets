# Natural Evolution Strategy (2008)

import numpy as np
import matplotlib.pyplot as plt

def create_fax_unless_specifeid(fax):
    if fax is None:
        return plt.subplots()
    return fax

class Rosen:
    def __init__(self):
        pass

    def __call__(self, pts):
        a = 2.0
        b = 100.0
        X, Y = pts[:, 0], pts[:, 1]
        f = (a - X) ** 2 + b * (Y - X**2)**2
        return f

    def show(self, fax=None):
        N = 100
        b_min = np.array([-3, -3])
        b_max = - b_min
        xlin, ylin = [np.linspace(b_min[i], b_max[i], N) for i in range(2)]
        X, Y = np.meshgrid(xlin, ylin)
        pts = np.array(list(zip(X.flatten(), Y.flatten())))
        Z = self.__call__(pts).reshape((N, N))
        fig, ax = create_fax_unless_specifeid(fax)
        ax.contourf(X, Y, Z, levels=[2**n for n in range(17)])

fun = Rosen()
fun.show()
plt.show()
