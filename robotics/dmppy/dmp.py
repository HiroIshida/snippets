import numpy as np

class DMP(object):
    def __init__(self):
        # Transformation system
        self.alpha = 25.0             # = D = 20.0
        self.beta = self.alpha / 4.0  # = K / D = 100.0 / 20.0 = 5.0
        # Canonical system
        self.alpha_t = self.alpha / 3.0

    def phase(self, n_steps, t=None):
        phases = np.exp(-self.alpha_t * np.linspace(0, 1, n_steps))
        if t is None:
            return phases
        else:
            return phases[t]

    def spring_damper(self, x0, g, tau, s, X, Xd):
        return self.alpha * (self.beta * (g - X) - tau * Xd) / tau ** 2

    def forcing_term(self, x0, g, tau, w, s, X, scale=False):
        n_features = w.shape[1]
        f = np.dot(w, self._features(tau, n_features, s))
        if scale:
            f *= g - x0

        if X.ndim == 3:
            F = np.empty_like(X)
            F[:, :] = f
            return F
        else:
            return f

    def _features(self, tau, n_features, s):
        if n_features == 0:
            return np.array([])
        elif n_features == 1:
            return np.array([1.0])
        c = self.phase(n_features)
        h = np.diff(c)
        h = np.hstack((h, [h[-1]]))
        phi = np.exp(-h * (s - c) ** 2)
        return s * phi / phi.sum()

    def imitate(self, X, tau, n_features):
        n_steps, n_dims = X.shape
        dt = tau / float(n_steps - 1)
        g = X[:, -1]

        Xd = np.vstack((np.zeros((1, n_dims)), np.diff(X, axis=0) / dt))
        Xdd = np.vstack((np.zeros((1, n_dims)), np.diff(Xd, axis=0) / dt))
        print Xdd

        F = tau * tau * Xdd - self.alpha * (self.beta * (g[:, np.newaxis] - X)
                                            - tau * Xd)

        design = np.array([self._features(tau, n_features, s)
                           for s in self.phase(n_steps)])
        #w = np.linalg.lstsq(design, F)[0].T
        from sklearn.linear_model import Ridge
        lr = Ridge(alpha=1.0, fit_intercept=False)
        lr.fit(design, F)
        w = lr.coef_

        return w

t_seq = np.linspace(0, 1.0, 100)
x_seq = np.vstack((np.sin(t_seq), np.cos(t_seq))).T
tau = 1.0
dmp = DMP()
w = dmp.imitate(x_seq, 1.0, 5)


"""
def trajectory(dmp, w, x0, g, tau, dt, o=None, shape=True, avoidance=False,
               verbose=0):
    if verbose >= 1:
        print("Trajectory with x0 = %s, g = %s, tau=%.2f, dt=%.3f"
              % (x0, g, tau, dt))

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
    S = dmp.phase(n_internal_steps + 1)
    while t < tau:
        t += internal_dt
        ti += 1
        s = S[ti]

        x += internal_dt * xd
        xd += internal_dt * xdd

        sd = dmp.spring_damper(x0, g, tau, s, x, xd)
        f = dmp.forcing_term(x0, g, tau, w, s, x) if shape else 0.0
        C = dmp.obstacle(o, x, xd) if avoidance else 0.0
        xdd = sd + f + C

        if ti % steps_between_measurement == 0:
            X.append(x.copy())
            Xd.append(xd.copy())
            Xdd.append(xdd.copy())

    return np.array(X), np.array(Xd), np.array(Xdd)
"""
