from sklearn.metrics.pairwise import *
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import numpy.random as rn
import cvxopt
import math
rn.seed(5)

def show2d(func, bmin, bmax, N = 20, fax = None, levels = None):
    # fax is (fix, ax)
    # func: np.array([x1, x2]) list -> scalar
    # func cbar specifies the same height curve
    if fax is None:
        fig, ax = plt.subplots() 
    else:
        fig = fax[0]
        ax = fax[1]

    mat_contour_ = np.zeros((N, N))
    mat_contourf_ = np.zeros((N, N))
    x1_lin, x2_lin = [np.linspace(bmin[i], bmax[i], N) for i in range(bmin.size)]
    for i in range(N):
        for j in range(N):
            x = np.array([x1_lin[i], x2_lin[j]])
            val_c, val_cf = func(x)
            mat_contour_[i, j] = val_c
            mat_contourf_[i, j] = val_cf
    mat_contour = mat_contour_.T
    mat_contourf = mat_contourf_.T
    X, Y = np.meshgrid(x1_lin, x2_lin)

    cs = ax.contour(X, Y, mat_contour, levels = levels, cmap = 'jet')
    zc = cs.collections[0]
    plt.setp(zc, linewidth=4)
    ax.clabel(cs, fontsize=10)
    cf = ax.contourf(X, Y, mat_contourf, cmap = 'gray_r')
    fig.colorbar(cf)


def gen_dataset(N):
    predicate = lambda x: x[0]**2 + x[1]**2 < 0.5 ** 2
    xp_lst = []
    xm_lst = []
    for i in range(N):
        x = rn.random(2) * 2 + np.array([-1, -1])
        if predicate(x): 
            xp_lst.append(x)
        else:
            xm_lst.append(x)
    n_p = len(xp_lst)
    yp_lst = [+1 for i in range(len(xp_lst))]
    ym_lst = [-1 for i in range(len(xm_lst))]
    x_lst = xp_lst + xm_lst
    y_lst = yp_lst + ym_lst

    X = np.zeros((2, N))
    for i in range(N):
        X[:,i] = x_lst[i]
    Y = np.array(y_lst)
    return X, Y, n_p

def gen_diadig(vec):
    n = vec.size
    Y = np.array([vec])
    tmp = np.repeat(Y, n, 0)
    ymat = tmp * tmp.T
    return ymat

class SSLM:

    def __init__(self, X, y, kern, nu = 0.1, nu1 = 0.01, nu2 = 0.02):
        self.X = X
        self.y = y
        self.kern = kern
        self.nu = nu
        self.nu1 = nu1 
        self.nu2 = nu2

        self.N = len(y)
        self.m1 = (self.N + sum(y))/2
        self.m2 = N - self.m1

        self.R, self.rho, self.cc, self.a_lst, self.idxes_SVp, self.idxes_SVn\
                = self._compute_important_parameters()

    def predict(self, x):
        val = (self.R**2 - self.cc - self.kern(x, x) + sum(2 * self.kern(X.T, x).flatten() * self.a_lst * self.y)).item()
        return val

    def _compute_important_parameters(self):
        gram, gramymat, a_lst = self._solve_qp()

        eps = 0.00001
        idxes_S1 = []
        for i in range(self.m1):
            a = a_lst[i]
            if eps < a and a < 1.0/(self.nu1 * self.m1) - eps:
                idxes_S1.append(i)

        idxes_S2 = []
        for i in range(self.m1, self.m1 + self.m2):
            a = a_lst[i]
            if eps < a and a < 1.0/(self.nu2 * self.m2) - eps:
                idxes_S2.append(i)

        cc = sum(sum(gen_diadig(a_lst) * gramymat))
        f_inner = lambda idx: gram[idx, idx] - sum(2 * gram[idx, :] * a_lst * self.y) + cc

        P1 = sum(map(f_inner, idxes_S1))
        P2 = sum(map(f_inner, idxes_S2))

        n1 = len(idxes_S1)
        n2 = len(idxes_S2)
        R = math.sqrt(P1/n1)
        rho = math.sqrt(P2/n2 - P1/n1)
        return R, rho, cc, a_lst, idxes_S1, idxes_S2

    def _solve_qp(self):
        Ymat = gen_diadig(np.array(self.y))

        gram = kern(self.X.T)
        gram_diag_matrix = np.diag(gram)

        gramymat = gram * Ymat
        gramymat_diag = np.array([-gram_diag_matrix]).T * np.array([self.y]).T

        P = cvxopt.matrix(gramymat)
        q = cvxopt.matrix(gramymat_diag)

        # eq 15
        A_15 = np.array([self.y],dtype = np.float64)
        b_15 = np.eye(1)
        A_16 = np.ones((1, self.N))
        b_16 = np.eye(1)*(2 * self.nu + 1)
        A_ = np.vstack((A_15, A_16))
        B_ = np.vstack((b_15, b_16))
        A = cvxopt.matrix(A_)
        b = cvxopt.matrix(B_)

        G0 = np.eye(self.N)
        G1 = - np.eye(self.N)
        G_ = np.vstack((G0, G1))
        G = cvxopt.matrix(G_)

        h0p = np.ones(self.m1)/(self.nu1 * self.m1)
        h0m = np.ones(self.N - self.m1)/(self.nu2 * (self.N - self.m1))
        h1 = np.zeros(self.N)
        h_ = np.block([h0p, h0m, h1])
        h = cvxopt.matrix(h_)
        sol = cvxopt.solvers.qp(P, q, A=A, b=b, G=G, h=h)
        a_lst = np.array([sol["x"][i] for i in range(self.N)])
        return gram, gramymat, a_lst

if __name__ == '__main__':
    N = 100
    X, y, n_p = gen_dataset(N)
    kern = linear_kernel

    sslm = SSLM(X, y, kern)
    sslm.predict([0, 0.8])


    ## plot

    fig, ax = plt.subplots() 


    bmin = np.array([-1, -1])
    bmax = np.array([1, 1])
    def f(x):
        tmp = sslm.predict(x)
        return tmp, tmp
    show2d(f, bmin, bmax, fax = (fig, ax))

    idx_positive = np.where(y == 1)
    idx_negative = np.where(y == -1)

    plt.scatter(X[0, idx_positive], X[1, idx_positive], c = "blue")
    plt.scatter(X[0, idx_negative], X[1, idx_negative], c = "red")

    plt.scatter(X[0, sslm.idxes_SVp], X[1, sslm.idxes_SVp], c = "blue", marker = 'v', s = 100)
    plt.scatter(X[0, sslm.idxes_SVn], X[1, sslm.idxes_SVn], c = "red", marker = 'v', s = 100)

    plt.show()


