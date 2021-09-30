
from sklearn.metrics.pairwise import *
import numpy as np
import numpy.linalg as la
import numpy.random as rn
import cvxopt

n_train = 30
X = rn.random((2, n_train))
gram = linear_kernel(X.T)
gram_diag_matrix = np.diag(gram)

P = cvxopt.matrix(gram)
q = cvxopt.matrix(np.array([-gram_diag_matrix]).T)

A = cvxopt.matrix(np.ones((1, n_train)))
b = cvxopt.matrix(np.eye(1))

G0 = np.eye(n_train)
G1 = - np.eye(n_train)
G = cvxopt.matrix(np.vstack((G0, G1)))

C = 1.0/n_train
h0 = np.ones(n_train) * C
h1 = np.zeros(n_train)
h = cvxopt.matrix(np.block([h0, h1]))

sol = cvxopt.solvers.qp(P, q, A=A, b=b, G=G, h=h)

def isValid(x):
    eps = 0.0001
    if x < eps: 
        return False
    if x > C - eps:
        return False
    return True


xs = [sol["x"][i] for i in range(n_train)]
idx = np.where(isValid(xs))[0]

print(idx)

