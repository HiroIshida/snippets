import numpy as np
from copy import copy

def generate_pair(dim):
    for i in range(dim):
        for j in range(0, i+1, 1):
            yield (i, j)

def calculate_dim(theta):
    L = len(theta)
    return int((-1 + np.sqrt(1 + 8*L)) * 0.5)

def compute(theta):
    dim = calculate_dim(theta)
    pair_lst = list(generate_pair(dim))

    L = np.zeros((dim, dim))
    for ((i, j), val) in zip(pair_lst, theta):
        L[i, j] = val
    L_inv = np.linalg.inv(L)
    cov_inv = L_inv.T.dot(L_inv)

    I_tril = np.zeros((dim, dim)) # lower triangle of fisher info mat
    for (i, j) in generate_pair(dim):
        def cov_partderiv(idx):
            L_copied = copy(L)
            i_, j_ = pair_lst[idx]
            L_copied[i_, j_] = 1.0
            cov_deriv = L_copied.dot(L.T) + L.dot(L_copied.T)
            return cov_deriv

        cov_deriv_i, cov_deriv_j  = cov_partderiv(i), cov_partderiv(j)
        M = cov_inv.dot(cov_deriv_i).dot(cov_inv).dot(cov_deriv_j)
        I_tril[i, j] = 0.5 * np.trace(M)

    def symmetrize(A):
        return A + A.T - np.diag(A.diagonal())
    I = symmetrize(I_tril) # fisher info mat
    print(I)

compute([1, 0.0, 1])
