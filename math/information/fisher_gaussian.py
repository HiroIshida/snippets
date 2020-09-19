import numpy as np
from copy import copy
from scipy.stats import multivariate_normal

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

def cov(theta):
    dim = calculate_dim(theta)
    pair_lst = list(generate_pair(dim))
    L = np.zeros((dim, dim))
    for ((i, j), val) in zip(pair_lst, theta):
        L[i, j] = val
    M = L.dot(L.T)
    return M

def normpdf(x, theta):
    M = cov(theta)
    dim = M.shape[0]
    return multivariate_normal.pdf(x, np.zeros(dim), M)

def compute_grad(fun, x):
    dim = len(x)
    eps = 1e-3
    f0 = fun(x)
    grad = np.zeros(dim)
    for i in range(dim):
        x_ = copy(x)
        x_[i] += eps
        f1 = fun(x_)
        grad[i] = (f1 - f0)/eps
    return grad

def compute_mc(theta):
    dim = calculate_dim(theta)
    pdf = lambda x: normpdf(x, theta)
    M = cov(theta)

    N_mc = 10000
    X = np.random.multivariate_normal(np.zeros(dim), M, N_mc)

    def rank1_mat(x):
        fun = lambda theta: normpdf(x, theta)
        vec = compute_grad(fun, theta)/pdf(x) # grad_loglikeli
        return np.outer(vec, vec)
    rank1_mat_lst = [rank1_mat(x) for x in X]
    I_mc = sum(rank1_mat_lst) * (1.0/N_mc)
    print(I_mc)

compute_mc([1, 0.5, 1])
