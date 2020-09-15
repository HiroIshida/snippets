# Natural Evolution Strategy, IJML (2014) algorithm 5

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

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

class NaturalEvolution:
    def __init__(self, x_init):
        self.x_mean = x_init
        self.cov = np.diag((1, 1)) 
        self.A = np.linalg.cholesky(self.cov).T
        self.lam = 10
        self.n_dim = len(x_init)

    def step(self, fun):
        cov_inv = np.linalg.inv(self.cov)

        x_rands = np.random.multivariate_normal(self.x_mean, self.cov, self.lam)
        costs = fun(x_rands)

        diffs = [(x_rand - self.x_mean).reshape(1, self.n_dim) for x_rand in x_rands]
        nabla_x_lst = [diff.dot(cov_inv) for diff in diffs]
        nabla_cov_lst = [0.5 * cov_inv.dot(np.outer(diff, diff)).dot(cov_inv) - 0.5 * cov_inv \
                for diff in diffs]

        nabla_A_lst = [self.A.dot(nabla_cov + nabla_cov.T).reshape((1, self.n_dim**2)) for nabla_cov in nabla_cov_lst]
        print(nabla_A_lst)

        # making a big Phi matrix
        left = np.vstack(diffs)
        middle = np.vstack(nabla_A_lst)
        right = np.ones((self.lam, 1))
        Phi = np.hstack((left, middle, right))
        print(np.linalg.matrix_rank(Phi))

        Phi_psudo_inv = np.linalg.inv(Phi.T.dot(Phi)).dot(Phi.T)
        #R = costs.reshape((self.lam, 1))

#        d_theta = Phi_psudo_inv.dot(R)




        




fun = Rosen()
nes = NaturalEvolution(np.ones(2))
nes.step(fun)
#fun.show()
#plt.show()
