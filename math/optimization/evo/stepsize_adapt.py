import numpy as np
import matplotlib.pyplot as plt

def create_fax_unless_specifeid(fax):
    if fax is None:
        return plt.subplots()
    return fax

class Rosen:
    def __init__(self):
        pass

    def __call__(self, pts):
        if pts.ndim == 1:
            pts = np.array([pts])
        a = 2.0
        b = 100.0
        X, Y = pts[:, 0], pts[:, 1]
        f = (a - X) ** 2 + b * (Y - X**2)**2
        #f = X ** 2 + Y ** 2
        return f

    def show(self, fax=None):
        N = 100
        b_min = np.array([-3, -3])
        b_max = - b_min
        xlin, ylin = [np.linspace(b_min[i], b_max[i], N) for i in range(2)]
        X, Y = np.meshgrid(xlin, ylin)
        pts = np.array(list(zip(X.flatten(), Y.flatten())))
        Z = self.__call__(pts).reshape((N, N))
        fig, ax = create_fax_unless_specifeid(fax)
        #ax.contourf(X, Y, Z, levels=[2**n for n in range(17)])
        ax.contourf(X, Y, Z)

class Optimizer(object):
    def __init__(self, x_init, fun, sigma=0.5):
        self.x = x_init
        self.n = len(x_init)
        self.sigma = sigma
        self.fun = fun
        self.N_mc = 20

        # see hansen, ., auger 2015 evolution strategies Algorithm 4
        self.mu = int(self.N_mc * 0.25)
        self.d = 3 * self.n
        self.s = 0 
        self.c_sigma = np.sqrt(self.mu / ((self.n + self.mu) * 1.0))

    def step(self):
        x_rands = np.array([self.x + np.random.randn(self.n) * self.sigma 
            for _ in range(self.N_mc)])
        fs = self.fun(x_rands)
        idxes_elite = np.argsort(fs)[:self.mu]
        x_elites = x_rands[idxes_elite, :]
        self.x = np.mean(x_elites, axis=0)

func = Rosen()
opt = Optimizer(np.array([5.0, 5.0]), func)
path = []
for i in range(300):
    opt.step()
    path.append(opt.x)
P = np.vstack(path)
fig, ax = plt.subplots()
func.show((fig, ax))
ax.plot(P[:, 0], P[:, 1])
plt.show()
