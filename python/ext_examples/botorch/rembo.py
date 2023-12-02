import tqdm
from dataclasses import dataclass
import warnings
import torch
import numpy as np
from torch import Tensor
from botorch.acquisition import UpperConfidenceBound
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils import standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from typing import List, Union, Optional


@dataclass
class Bound:
    bmin: np.ndarray
    bmax: np.ndarray

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.bmin) / (self.bmax - self.bmin)

    def de_normalize(self, x: np.ndarray) -> np.ndarray:
        return x * (self.bmax - self.bmin) + self.bmin

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.bmin, self.bmax)

    def axis_projection(self, x: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(x, self.bmin), self.bmax)



class BayesOpt:
    X: List[np.ndarray] 
    Y: List[float]
    gp: SingleTaskGP
    opt_param_history: List[Tensor]
    opt_value_history: List[float]
    y_mean: float
    y_std: float
    bound: Bound

    def fit_gp(self, X: List[np.ndarray], Y: List[float]) -> SingleTaskGP:
        X_normalized = [self.bound.normalize(x) for x in X]
        X_ = torch.Tensor(np.vstack(X_normalized))
        Y_ = standardize(torch.Tensor(Y).unsqueeze(-1))
        gp = SingleTaskGP(X_, Y_)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)
        warnings.filterwarnings("ignore", message="The model inputs are of type torch.float32")
        warnings.filterwarnings("ignore", message="gen_candidates_scipy")
        return gp

    def __init__(self, X: List[np.ndarray], Y: List[float], bmin: np.ndarray, bmax: np.ndarray):
        self.X = X
        self.Y = Y
        self.bound = Bound(bmin, bmax)
        self.gp = self.fit_gp(X, Y)
        self.opt_param_history = []
        self.opt_value_history = []

    @property
    def dim(self) -> int:
        return len(self.X[0])

    def ask_init(self) -> np.ndarray:
        return np.random.uniform(self.bound.bmin, self.bound.bmax)

    def ask(self) -> np.ndarray:
        UCB = UpperConfidenceBound(self.gp, beta=0.1)
        bounds = torch.stack([Tensor(np.zeros(self.dim)), Tensor(np.ones(self.dim))])
        candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
        )
        self.opt_param_history.append(candidate)
        self.opt_value_history.append(float(acq_value.item()))
        x_normalized = candidate.numpy()
        x = self.bound.de_normalize(x_normalized)
        return x

    def tell(self, x: np.ndarray, y: float) -> None:
        self.X.append(x)
        self.Y.append(y)
        self.gp = self.fit_gp(self.X, self.Y)


class RandomEmbeddingBayesOpt:
    bo_inner: Optional[BayesOpt]
    M_random: np.ndarray
    _z_list: List[np.ndarray]
    _z_bound: Bound
    _x_bound: Bound

    def __init__(self, bmin: np.ndarray, bmax: np.ndarray, dim_intrinsic: int = 3):
        dim = len(bmin)
        M = np.random.randn(dim, dim_intrinsic)
        self.M_random = M
        self._z_list = []
        bound_scaling = np.sqrt(dim_intrinsic)  # folloiwing the paper
        self._x_bound = Bound(bmin, bmax)
        self._z_bound = Bound(-bound_scaling * np.ones(dim_intrinsic), bound_scaling * np.ones(dim_intrinsic))
        self.bo_inner = None

    def z_to_x(self, z: np.ndarray) -> np.ndarray:
        xn = z.dot(self.M_random.T).flatten()
        x = self._x_bound.de_normalize(xn)
        x_projected = self._x_bound.axis_projection(x)
        return x_projected

    def _find_corresponding_z(self, x: np.ndarray) -> np.ndarray:
        for z in self._z_list:
            x_hash = self.z_to_x(z)
            max_diff = np.max(np.abs(x_hash - x))
            if max_diff < 1e-5:
                return z
        assert False

    def ask_init(self) -> np.ndarray:
        z_rand = self._z_bound.sample()
        self._z_list.append(z_rand)
        x_rand = self.z_to_x(z_rand)
        return x_rand

    def ask(self) -> np.ndarray:
        assert self.bo_inner is not None
        z = self.bo_inner.ask()
        self._z_list.append(z)
        return self.z_to_x(z)

    def tell(self, x: np.ndarray, y: float) -> None:
        z = self._find_corresponding_z(x)
        if self.bo_inner is None:
            self.bo_inner = BayesOpt([z], [y], self._z_bound.bmin, self._z_bound.bmax)
        else:
            self.bo_inner.tell(z, y)


def f_bench(x):
    t1 = np.sum(x ** 4)
    t2 = - 16 * np.sum(x ** 2)
    t3 = 5 * np.sum(x)
    return -0.5 * (t1 + t2 + t3)


n_dim = 10
train_X = [np.ones(n_dim) * 5]
train_Y = [f_bench(x) for x in train_X]
# bo = BayesOpt(train_X, train_Y, -5 * np.ones(n_dim), 5 * np.ones(n_dim))
bo = RandomEmbeddingBayesOpt(-5 * np.ones(n_dim), 5 * np.ones(n_dim))
# bo = RandomEmbeddingBayesOpt(np.zeros(n_dim), np.ones(n_dim))
for _ in range(5):
    x = bo.ask_init()
    y = f_bench(x)
    bo.tell(x, y)

for _ in tqdm.tqdm(range(300)):
    x = bo.ask()
    y = f_bench(x)
    bo.tell(x, y)
    print(y)
