import numpy as np

class CubicSpline(object):
    def __init__(self, x, y):
        n_piece = len(x) - 1
        n_coef = n_piece * 4

        # try to determine w by solving linear equation A*w = b
        A = np.zeros((n_coef, n_coef))
        b = np.zeros(n_coef)

        zeros = np.zeros(len(x))
        ones = np.ones(len(x))
        xx = x**2
        xxx = x**3

        phi = np.vstack([ones, x, xx, xxx]).T
        phi_d = np.vstack([zeros, ones, 2 * x, 3 * xx]).T
        phi_dd = np.vstack([zeros, zeros, 2 * ones, 6 * x]).T

        idx = 0

        # 0-th order condition 
        for i in range(n_piece):
            # left
            A[idx, 4*i:4*(i+1)] = phi[i, :]
            b[idx] = y[i]
            idx += 1

            # right
            A[idx, 4*i:4*(i+1)] = phi[i+1, :]
            b[idx] = y[i+1]
            idx += 1

        # 1-st order condition
        for i in range(n_piece-1):
            A[idx, 4*i:4*(i+1)] = phi_d[i+1, :]
            A[idx, 4*(i+1):4*(i+2)] = -phi_d[i+1, :]
            idx += 1

        # 2-nd order condition
        for i in range(n_piece-1):
            A[idx, 4*i:4*(i+1)] = phi_dd[i+1, :]
            A[idx, 4*(i+1):4*(i+2)] = -phi_dd[i+1, :]
            idx += 1

        # natural spline condition for start
        A[idx, 0:4] = phi_dd[0, :]
        idx += 1

        # natural spline condition for end
        A[idx, -4:] = phi_dd[n_piece, :]
        idx += 1

        w = np.linalg.solve(A, b)

        self.x = x
        self.w = w

    def __call__(self, x_query):
        i = min(np.where(x_query < self.x))[0] - 1
        return np.dot(self.w[i * 4: (i+1) * 4], 
                np.array([1, x_query, x_query**2, x_query**3]))


if __name__=='__main__':
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1, 2, 3, 2, 1.5, 1.3, 1.1, 1.0, 0.9, 0.6])

    import time
    ts = time.time()
    for i in range(1000):
        itp = CubicSpline(x, y)
    print((time.time() - ts)/1000)

    y = itp(2.0)

    xs = np.linspace(0.1, 8.9, 100)
    ys = [itp(x) for x in xs]
    import matplotlib.pyplot as plt
    plt.plot(xs, ys)
    plt.show()





