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

        angle = 0.8
        c = np.cos(angle); s = np.sin(angle)
        mat = np.array([[c, s],[-s, c]])
        pts = pts.dot(mat)

        X, Y = pts[:, 0], pts[:, 1]
        #f = (a - X) ** 2 + b * (Y - X**2)**2
        f = X ** 2 * np.cos(angle)**2 + Y ** 2 * np.sin(angle)**2  * 0.4
        return f

    def show(self, fax=None):
        N = 20
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
    def __init__(self, x_init, fun, sigma=0.1):
        self.x = x_init
        self.n = len(x_init)
        self.sigma = sigma
        self.fun = fun
        self.N_mc = 20

        # see hansen, ., auger 2015 evolution strategies Algorithm 4
        self.mu = int(self.N_mc * 0.25)
        self.di = 3 * self.n
        self.d = 1 + np.sqrt(self.mu / self.n)
        self.s = np.zeros(self.n)
        self.c_sigma = np.sqrt(self.mu / ((self.n + self.mu) * 1.0)) * 1.0

    def step(self):
        rands = np.random.randn(self.N_mc, self.n) # z
        x_rands = np.array([self.x + rn * self.sigma for rn in rands])
        fs = self.fun(x_rands)
        idxes_elite = np.argsort(fs)[:self.mu]
        x_elites = x_rands[idxes_elite, :]
        self.x = np.mean(x_elites, axis=0)

        coef =  np.sqrt(self.c_sigma * (2 - self.c_sigma)) * np.sqrt(self.mu)/self.mu
        self.s = (1 - self.c_sigma) * self.s + coef * sum(rands[idxes_elite])
        self.sigma = self.sigma * np.exp(np.abs(self.s) / 1.0 - 1)**(1.0/self.di) \
                * np.exp(np.linalg.norm(self.s)/np.sqrt(self.n) - 1.0)**(self.c_sigma/self.d)



func = Rosen()
opt = Optimizer(np.array([0.0, 8.0]), func)
path = [opt.x]
for i in range(30):
    opt.step()
    path.append(opt.x)
P = np.vstack(path)
fig, ax = plt.subplots()
func.show((fig, ax))
ax.plot(P[:, 0], P[:, 1], "o-")
plt.show()
