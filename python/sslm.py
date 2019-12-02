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


def gen_dataset_(N):
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

def gen_dataset():
    X = np.array([
        [0.0, 0.0],
        [0.5, 1.0]
        ])
    y = np.array([1, -1])
    return X, y, 1

def gen_diadig(vec):
    n = vec.size
    Y = np.array([vec])
    tmp = np.repeat(Y, n, 0)
    ymat = tmp * tmp.T
    return ymat


N = 60
#X, y, n_p = gen_dataset()
X, y, n_p = gen_dataset_(N)
kern = linear_kernel

nu1 = 0.01
nu2 = 0.01
nu = 0.2

Ymat = gen_diadig(np.array(y))

gram = kern(X.T)
gram_diag_matrix = np.diag(gram)

gramymat = gram * Ymat
gramymat_diag = np.array([-gram_diag_matrix]).T * np.array([y]).T

P = cvxopt.matrix(gramymat)
q = cvxopt.matrix(gramymat_diag)

# eq 15
A_15 = np.array([y],dtype = np.float64)
b_15 = np.eye(1)
A_16 = np.ones((1, N))
b_16 = np.eye(1)*(2 * nu + 1)
A_ = np.vstack((A_15, A_16))
B_ = np.vstack((b_15, b_16))
A = cvxopt.matrix(A_)
b = cvxopt.matrix(B_)

G0 = np.eye(N)
G1 = - np.eye(N)
G_ = np.vstack((G0, G1))
G = cvxopt.matrix(G_)

h0p = np.ones(n_p)/(nu1 * n_p)
h0m = np.ones(N - n_p)/(nu2 * (N - n_p))
h1 = np.zeros(N)
h_ = np.block([h0p, h0m, h1])
h = cvxopt.matrix(h_)
sol = cvxopt.solvers.qp(P, q, A=A, b=b, G=G, h=h)
a_lst = np.array([sol["x"][i] for i in range(N)])

m1 = n_p
m2 = N - n_p

eps = 0.00001
idxes_S1 = []
for i in range(m1):
    a = a_lst[i]
    if eps < a and a < 1.0/(nu1 * m1) - eps:
        idxes_S1.append(i)

idxes_S2 = []
for i in range(m1, m1 + m2):
    a = a_lst[i]
    if eps < a and a < 1.0/(nu2 * m2) - eps:
        idxes_S2.append(i)

cc = sum(sum(gen_diadig(a_lst) * gramymat))
f_inner = lambda idx: gram[idx, idx] - sum(2 * gram[idx, :] * a_lst * y) + cc

P1 = sum(map(f_inner, idxes_S1))
P2 = sum(map(f_inner, idxes_S2))

n1 = len(idxes_S1)
n2 = len(idxes_S2)
R = math.sqrt(P1/n1)
rho = math.sqrt(P2/n2 - P1/n1)

Xt = X.T
x_new = Xt[1, :]
predict = lambda x: (R**2 - cc - kern(x, x) + sum(2 * kern(X.T, x).flatten() * a_lst * y)).item()
predict([0, 0.8])


## plot

fig, ax = plt.subplots() 


bmin = np.array([-1, -1])
bmax = np.array([1, 1])
def f(x):
    tmp = predict(x)
    return tmp, tmp
show2d(f, bmin, bmax, fax = (fig, ax))

idx_positive = np.where(y == 1)
idx_negative = np.where(y == -1)

plt.scatter(X[0, idx_positive], X[1, idx_positive], c = "blue")
plt.scatter(X[0, idx_negative], X[1, idx_negative], c = "red")

plt.scatter(X[0, idxes_S1], X[1, idxes_S1], c = "blue", marker = 'v', s = 100)
plt.scatter(X[0, idxes_S2], X[1, idxes_S2], c = "red", marker = 'v', s = 100)

plt.show()

