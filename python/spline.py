import numpy as np

if __name__=='__main__':
    def create_interp(x, y):
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

        def interp(a):
            i = min(np.where(a < x))[0] - 1
            return np.dot(w[i * 4: (i+1) * 4], np.array([1, a, a**2, a**3]))
        return interp

    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1, 2, 3, 2, 1.5, 1.3, 1.1, 1.0, 0.9, 0.6])
    itp = create_interp(x, y)
    y = itp(2.0)

    xs = np.linspace(0.1, 8.9, 100)
    ys = [itp(x) for x in xs]

    import matplotlib.pyplot as plt
    plt.plot(xs, ys)
    plt.show()





