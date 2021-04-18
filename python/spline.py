import numpy as np

class DesignMatrix(object):

    def __init__(self, n_piece):
        self.A = np.zeros((4 * n_piece, 4 * n_piece))
        self.b = np.zeros(4 * n_piece)
        self.idx = 0

    def set_Aval(self, j, vec):
        # i-th constraint, j-th piece
        self.A[self.idx, 4*j:4*(j+1)] = vec

    def set_bval(self, val):
        self.b[self.idx] = val

    def next(self):
        self.idx += 1

if __name__=='__main__':
    def create_interp(x, y):
        n_piece = len(x) - 1
        n_coef = n_piece * 4
        M = DesignMatrix(n_piece)

        zeros = np.zeros(len(x))
        ones = np.ones(len(x))
        xx = x**2
        xxx = x**3
        phi = np.vstack([ones, x, xx, xxx]).T
        phi_d = np.vstack([zeros, ones, 2 * x, 3 * xx]).T
        phi_dd = np.vstack([zeros, zeros, 2 * ones, 6 * x]).T

        # 0-th order condition 
        for i in range(n_piece):
            M.set_Aval(i, phi[i, :])
            M.set_bval(y[i])
            M.next()

            M.set_Aval(i, phi[i+1, :])
            M.set_bval(y[i+1])
            M.next()

        for i in range(n_piece-1):
            M.set_Aval(i, phi_d[i+1, :])
            M.set_Aval(i+1, -phi_d[i+1, :])
            M.set_bval(0)
            M.next()

        for i in range(n_piece-1):
            M.set_Aval(i, phi_dd[i+1, :])
            M.set_Aval(i+1, -phi_dd[i+1, :])
            M.set_bval(0)
            M.next()

        # natural spline condition for start
        M.set_Aval(0, phi_dd[0, :])
        M.set_bval(0)
        M.next()

        # natural spline condition for end
        M.set_Aval(n_piece-1, phi_dd[n_piece, :])
        M.set_bval(0)
        M.next()

        w = np.linalg.inv(M.A).dot(M.b)

        def interp(a):
            idx = min(np.where(a < x))[0] - 1
            return np.dot(w[idx * 4: (idx+1) * 4], np.array([1, a, a**2, a**3]))
        return interp

    x = np.array([0, 1, 2, 3, 4])
    y = np.array([1, 2, 3, 2, 1.5])
    itp = create_interp(x, y)

    xs = np.linspace(0.1, 3.9, 100)
    ys = [itp(x) for x in xs]

    import matplotlib.pyplot as plt
    plt.plot(xs, ys)
    plt.show()




        












