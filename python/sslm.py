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


def gen_ymat(y):
    n = len(y)
    Y = np.array([y])
    tmp = np.repeat(Y, n, 0)
    ymat = tmp * tmp.T
    return ymat

def solve_sslm(X, y, N, n_p):
    nu1 = 0.1
    nu2 = 0.1
    nu = 0.2

    Ymat = gen_ymat(y)

    gram = linear_kernel(X.T)
    gram_diag_matrix = np.diag(gram)

    P = cvxopt.matrix(gram * Ymat)
    q = cvxopt.matrix(
            - np.array([-gram_diag_matrix]).T * 
            np.array([y]).T)

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
    return sol

N = 100
X, y, n_p = gen_dataset(N)

sol = solve_sslm(X, y, N, n_p)




xs = [sol["x"][i] for i in range(N)]

