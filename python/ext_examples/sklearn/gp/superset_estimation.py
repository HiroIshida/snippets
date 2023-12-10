import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

def func(x):
    # max of two quadratic functions
    f1 = -(x + 0.75) ** 2 + 1.0
    f2 = -(x - 0.75) ** 2 + 1.0
    return max(f1, f2).item() - 0.85


class ActiveLevelsetEstimation:
    # Bryan, Brent, et al. "Active learning for identifying function threshold boundaries." Advances in neural information processing systems 18 (2005).
    gp: GaussianProcessRegressor
    n_sampler_per_ask: int
    n_budget_random_search: int
    b_min: np.ndarray
    b_max: np.ndarray
    X: np.ndarray
    Y: np.ndarray

    def __init__(self,
                 X: np.ndarray,
                 Y: np.ndarray,
                 ls: np.ndarray,
                 b_min: np.ndarray,
                 b_max: np.ndarray,
                 n_budget_random_search: int = 100,
                 n_sample_per_ask: int = 1):
        assert X.ndim == 2
        assert Y.ndim == 1

        rbf = RBF(ls)
        gp = GaussianProcessRegressor(kernel=rbf, optimizer=None) 
        gp.fit(X, Y)
        self.gp = gp
        self.n_sampler_per_ask = n_sample_per_ask
        self.n_budget_random_search = n_budget_random_search
        self.b_min = b_min
        self.b_max = b_max
        self.X = X
        self.Y = Y

    def ask(self) -> np.ndarray:
        X = np.random.uniform(self.b_min,
                              self.b_max,
                              size=(self.n_budget_random_search, self.b_min.shape[0]))
        Y, sigmas = self.gp.predict(X, return_std=True)
        acq_values = - np.abs(Y) + 1.96 * sigmas
        X = X[np.argsort(acq_values)[::-1]]
        return X[:self.n_sampler_per_ask]

    def tell(self, X: np.ndarray, Y: np.ndarray):
        self.X = np.concatenate([self.X, X])
        self.Y = np.hstack([self.Y, Y])
        self.gp.fit(self.X, self.Y)


if __name__ == "__main__":
    # set up the problem
    b_min = np.array([-1.7])
    b_max = np.array([2.0])
    X = np.array([[-1.0], [1.0]])
    Y = np.array([func(xi) for xi in X])
    ls = np.array([0.1])
    # set up the active learner
    ale = ActiveLevelsetEstimation(X, Y, ls, b_min, b_max, n_sample_per_ask=1)
    # run the active learning loop
    ts = time.time()
    for i in range(50):
        X = ale.ask()
        Y = np.array([func(xi) for xi in X])
        ale.tell(X, Y)
    print(f"Time: {time.time() - ts}")

    ts = time.time()
    ale.gp.predict(np.random.uniform(b_min, b_max, size=(10000, 1)))
    print(f"Time: {time.time() - ts}")


    # visualize
    xlin = np.linspace(-2, 2, 100)
    ys = np.array([func(x) for x in xlin])
    fig, ax = plt.subplots()
    ax.plot(xlin, ys)

    # gp with band
    y, sigmas = ale.gp.predict(xlin.reshape(-1, 1), return_std=True)
    ax.plot(xlin, y)
    ax.fill_between(xlin, y - 1.96 * sigmas, y + 1.96 * sigmas, alpha=0.2)
    ax.scatter(ale.X, ale.Y, c="red")
    plt.show()
