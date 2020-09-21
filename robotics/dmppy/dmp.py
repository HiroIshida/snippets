import numpy as np

class DMP(object):
    def __init__(self, w=None, g=None):
        # Transformation system
        self.alpha = 25.0             # = D = 20.0
        self.beta = self.alpha / 4.0  # = K / D = 100.0 / 20.0 = 5.0
        # Canonical system
        self.alpha_t = self.alpha / 3.0
        self.w = w
        self.g = g

    @classmethod
    def imitate(cls, X, tau, n_features):
        dmp = DMP()
        dmp.g = X[-1, :]
        dmp.w = dmp._imitate(X, tau, n_features)
        return dmp

    def phase(self, n_steps):
        phases = np.exp(-self.alpha_t * np.linspace(0, 1, n_steps))
        return phases

    def spring_damper(self, x0, g, tau, s, X, Xd):
        return self.alpha * (self.beta * (g - X) - tau * Xd) / tau ** 2

    def forcing_term(self, x0, g, tau, w, s, X):
        n_features = w.shape[1]
        f = np.dot(w, self._features(tau, n_features, s))
        return f

    def _features(self, tau, n_features, s):
        c = self.phase(n_features)
        h = np.diff(c)
        h = np.hstack((h, [h[-1]]))
        phi = np.exp(-h * (s - c) ** 2)
        return s * phi / phi.sum()

    def _imitate(self, X, tau, n_features):
        n_steps, n_dims = X.shape
        dt = tau / float(n_steps - 1)
        g = X[-1, :]

        Xd = np.vstack((np.zeros((1, n_dims)), np.diff(X, axis=0) / dt))
        Xdd = np.vstack((np.zeros((1, n_dims)), np.diff(Xd, axis=0) / dt))

        g_repeated = np.repeat(g[np.newaxis, :], n_steps, axis=0)
        F = tau * tau * Xdd - self.alpha * (self.beta * (g_repeated - X)
                                            - tau * Xd)

        design = np.array([self._features(tau, n_features, s)
                           for s in self.phase(n_steps)])
        #w = np.linalg.lstsq(design, F)[0].T
        from sklearn.linear_model import Ridge
        lr = Ridge(alpha=1.0, fit_intercept=False)
        lr.fit(design, F)
        w = lr.coef_

        return w

    def trajectory(self, x0, tau, dt):
        x = x0.copy()
        xd = np.zeros_like(x, dtype=np.float64)
        xdd = np.zeros_like(x, dtype=np.float64)
        X = [x0.copy()]
        Xd = [xd.copy()]
        Xdd = [xdd.copy()]

        # Internally, we do Euler integration usually with a much smaller step size
        # than the step size required by the system
        internal_dt = min(0.001, dt)
        n_internal_steps = int(tau / internal_dt)
        steps_between_measurement = int(dt / internal_dt)

        # Usually we would initialize t with 0, but that results in floating point
        # errors for very small step sizes. To ensure that the condition t < tau
        # really works as expected, we add a constant that is smaller than
        # internal_dt.
        t = 0.5 * internal_dt
        ti = 0
        S = self.phase(n_internal_steps + 1)
        while t < tau:
            t += internal_dt
            ti += 1
            s = S[ti]

            x += internal_dt * xd
            xd += internal_dt * xdd

            sd = self.spring_damper(x0, self.g, tau, s, x, xd)
            f = self.forcing_term(x0, self.g, tau, self.w, s, x) 
            xdd = sd + f 

            if ti % steps_between_measurement == 0:
                X.append(x.copy())
                Xd.append(xd.copy())
                Xdd.append(xdd.copy())

        return np.array(X), np.array(Xd), np.array(Xdd)


t_seq = np.linspace(0, 1.0, 100)
x_seq = np.vstack((np.sin(t_seq), np.cos(t_seq))).T
tau = 1.0
dmp = DMP.imitate(x_seq, 1.0, 5)
X, _, _ = dmp.trajectory(np.zeros(2), 1.0, 1e-2)

import matplotlib.pyplot as plt
plt.plot(X[:, 0], "ro")
plt.plot(X[:, 1], "ro")
plt.show()


