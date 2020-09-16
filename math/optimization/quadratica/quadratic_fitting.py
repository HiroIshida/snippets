import numpy as np

def fit_quadratic_model(X, Y):

    def pair_generator(n): # generate n*(n - 1)/2 pairs
        for i in range(n):
            for j in range(i+1, n, 1):
                yield (i, j)

    def construct_design_matrix(X):
        mat = np.zeros((n_data, 1 + 2 * n_dim + n_dim * (n_dim - 1) / 2))
        counter = 0

        # filling 1
        mat[:, 0] = np.ones(n_data)
        counter += 1

        # filling x_i 
        for i in range(n_dim):
            mat[:, counter] = X[:, i]
            counter += 1

        # filling x_i ** 2
        for i in range(n_dim):
            mat[:, counter] = X[:, i] ** 2
            counter += 1

        # filling 2 * x_i * x_j
        for (i, j) in pair_generator(n_dim):
            mat[:, counter] = 2 * X[:, i] * X[:, j]
            counter += 1
        return mat

    def symmetrize(A):
        return A + A.T - np.diag(A.diagonal())

    n_data, n_dim = X.shape
    phi = construct_design_matrix(X)
    theta = np.linalg.inv(phi.T.dot(phi)).dot(phi.T).dot(Y)
    f = theta[0]
    g = theta[1:n_dim+1]

    h_tmp = np.diag(theta[n_dim+1:n_dim+n_dim+1])
    for ((i, j), val) in zip(pair_generator(n_dim), theta[2*n_dim+1:]):
        h_tmp[i, j] = val
    h = symmetrize(h_tmp)
    return f, g, h


def fun(X):
    v = X[:, 0]**2 + X[:, 1]**2 * 4.0 + 3
    return v

n_sample = 10000
X = np.random.randn(n_sample, 2) 
Y = fun(X) + np.random.randn(n_sample)
f, g, h = fit_quadratic_model(X, Y)
