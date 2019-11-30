from sklearn.metrics.pairwise import *
import numpy as np
import numpy.linalg as la
import numpy.random as rn
import cvxopt

def gen_dataset(N):
    predicate = lambda x: x[0]**2 + x[1]**2 < 1.0
    xp_lst = []
    xm_lst = []
    for i in range(N):
        x = rn.random(2)
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


N = 100
X, y, n_p = gen_dataset(N)
kern = linear_kernel

nu1 = 0.2
nu2 = 0.2
nu = 0.4

Ymat = gen_diadig(np.array(y))

gram = kern(X.T)
gram_diag_matrix = np.diag(gram)

gramymat = gram * Ymat
gramymat_diag = np.array([-gram_diag_matrix]).T * np.array([y]).T

P = cvxopt.matrix(gramymat)
q = cvxopt.matrix(- gramymat_diag)

# eq 15
A_15 = np.array([y],dtype = np.float64)
b_15 = np.eye(1)
A_16 = np.ones((1, N))
b_16 = np.eye(1)*(2 * nu + 1)
A = cvxopt.matrix(np.vstack((A_15, A_16)))
b = cvxopt.matrix(np.vstack((b_15, b_16)))

G0 = np.eye(N)
G1 = - np.eye(N)
G = cvxopt.matrix(np.vstack((G0, G1)))

h0p = np.ones(n_p)/(nu1 * n_p)
h0m = np.ones(N - n_p)/(nu2 * (N - n_p))
h1 = np.zeros(N)
h = cvxopt.matrix(np.block([h0p, h0m, h1]))
sol = cvxopt.solvers.qp(P, q, A=A, b=b, G=G, h=h)
a_lst = np.array([sol["x"][i] for i in range(N)])

m1 = n_p
m2 = N - n_p

idxes_S1 = []
eps = 0.00001
for i in range(m1):
    a = a_lst[i]
    if  eps < a and a < 1.0/(nu1 * m1) - eps:
        idxes_S1.append(i)

idxes_S2 = []
for i in range(m1, m1 + m2):
    a = a_lst[i]
    if eps < a and a < 1.0/(nu2 * m2) - eps:
        idxes_S2.append(i)

cc = sum(sum(gen_diadig(a_lst) * gramymat))

# eq 19
P1_inner = (x)-> kern(x, x) - 2 * sum(sum(

print(idxes_S1)
print(idxes_S2)

#kern(rn.rand(10, 2), np.array([[1, 1]]))
