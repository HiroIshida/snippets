# Natural Evolution Strategy, IJML (2014) algorithm 5

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm, expm

def create_fax_unless_specifeid(fax):
    if fax is None:
        return plt.subplots()
    return fax

class Rosen:
    def __init__(self):
        pass

    def __call__(self, pts):
        if pts.ndim == 1:
            pts = np.array([pts])
        X, Y = pts[:, 0], pts[:, 1]
        #a = 2.0
        #b = 100.0
        #f = (a - X) ** 2 + b * (Y - X**2)**2
        r = 9 * X ** 2 + Y**2
        return -np.exp(-r **2 * 0.1) # normal dist
        #f = X ** 2 + Y ** 2
        #return f

    def show(self, fax=None):
        N = 100
        b_min = np.array([-3, -3])
        b_max = - b_min
        xlin, ylin = [np.linspace(b_min[i], b_max[i], N) for i in range(2)]
        X, Y = np.meshgrid(xlin, ylin)
        pts = np.array(list(zip(X.flatten(), Y.flatten())))
        Z = self.__call__(pts).reshape((N, N))
        fig, ax = create_fax_unless_specifeid(fax)
        #ax.contourf(X, Y, Z, levels=[2**n for n in range(17)])
        ax.contourf(X, Y, Z)

class NaturalEvolution:
    def __init__(self, x_init):
        self.x_mean = x_init
        self.n_dim = len(x_init)

        self.cov = np.diag((1, 1)) * 0.01
        A = np.linalg.cholesky(self.cov).T
        self.sigma = abs(np.linalg.det(A)) ** (1/self.n_dim)
        self.B = A/self.sigma

        self.lam = 100
        self.eta_sigma = 3*(3+np.log(self.n_dim))*(1.0/(5*self.n_dim*np.sqrt(self.n_dim)))  
        self.eta_bmat = (9+3*np.log(self.n_dim))*(1.0/(5*self.n_dim*np.sqrt(self.n_dim)))  
        self.eta_mu = 1.0

    def step(self, fun):
        s_lst = np.random.randn(self.lam, self.n_dim)
        z_lst = self.x_mean + self.sigma * np.dot(s_lst, self.B)

        self.ptcl_cache = z_lst

        f_lst = [fun(z).item() for z in z_lst]

        f_mean = np.mean(f_lst)

        idxes_upper = np.where(f_lst - f_mean < 0)[0]
        n_nonzero = len(idxes_upper)
        u_lst = 1.0/float(n_nonzero) * (f_lst - f_mean < 0) 

        # compute nabla_delta 
        nabla_delta = sum([u * s for u, s in zip(u_lst, s_lst)])

        # compute nabla_mat
        tmp_lst = [u * (np.outer(s, s) - np.eye(self.n_dim)) for u, s in zip(u_lst, s_lst)]
        nabla_M = sum(tmp_lst)

        # compute nabla_sigma
        nabla_sigma = np.trace(nabla_M) / self.n_dim
        nabla_B = nabla_M  - nabla_sigma * np.eye(self.n_dim)

        self.x_mean += self.eta_mu * self.sigma * nabla_delta.dot(self.B.T)
        self.sigma *= np.exp(self.eta_sigma * 0.5 * nabla_sigma)
        self.B = self.B.dot(expm(self.eta_bmat * 0.5 * nabla_B))

fun = Rosen()
fun.show()
plt.show()
nes = NaturalEvolution(np.array([2.0, 2.0]))
for i in range(100):
    nes.step(fun)
    fig, ax = plt.subplots()
    fun.show((fig, ax))
    ax.scatter(nes.ptcl_cache[:, 0], nes.ptcl_cache[:, 1], c="r")
    plt.show()
    print(nes.x_mean)
