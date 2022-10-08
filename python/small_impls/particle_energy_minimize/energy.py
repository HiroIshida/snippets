import numpy as np
from scipy.optimize import OptimizeResult, minimize, Bounds
import matplotlib.pyplot as plt
import copy
import time
from typing import Callable, Tuple


def compute_distance_matrix(points: np.ndarray):
    assert points.ndim == 2
    n_point, n_dim = points.shape
    squared_dist_matrix = np.zeros((n_point, n_point))
    for i, p in enumerate(points):
        squared_dist_matrix[:, i] = np.sum((p - points) ** 2, axis=1)
    dist_matrix = np.sqrt(squared_dist_matrix)
    return dist_matrix


def fun_energy(points: np.ndarray, n_power=2):
    n_point, n_dim = points.shape

    dist_matrix = compute_distance_matrix(points)

    modified_dist_matrix = copy.deepcopy(dist_matrix)
    for i in range(n_point):
        modified_dist_matrix[i, i] = 1e5
    energy = 0.5 * np.sum(1.0 / (modified_dist_matrix ** n_power))  # 0.5 becaue double count

    part_grad_list = []
    for i, p in enumerate(points):
        diff = points - p
        r = modified_dist_matrix[:, i]
        tmp = (1.0 / r **(n_power + 2))
        part_grad = np.sum(n_power * np.tile(tmp, (n_dim, 1)).T * diff, axis=0)
        part_grad_list.append(part_grad)
    grad = np.hstack(part_grad_list)
    return energy, grad




def scipinize(fun: Callable) -> Tuple[Callable, Callable]:
    closure_member = {"jac_cache": None}

    def fun_scipinized(x):
        f, jac = fun(x)
        closure_member["jac_cache"] = jac
        return f

    def fun_scipinized_jac(x):
        return closure_member["jac_cache"]

    return fun_scipinized, fun_scipinized_jac


def gradient_test(func, x0, decimal=4):
    f0, grad = func(x0)
    n_dim = len(x0)

    eps = 1e-7
    grad_numerical = np.zeros(n_dim)
    for idx in range(n_dim):
        x1 = copy.copy(x0)
        x1[idx] += eps
        f1, _ = func(x1)
        grad_numerical[idx] = (f1 - f0) / eps

    print(grad_numerical)
    print(grad)
    np.testing.assert_almost_equal(grad, grad_numerical, decimal=decimal)

n_dim = 2
n_point = 30
points = np.random.rand(n_point, n_dim)
a = 2.5

def obj_fun(points: np.ndarray):
    f1, grad1 = fun_energy(points, n_power=1)
    f2, grad2 = fun_energy(points, n_power=-2)
    #return a * f1 + b * f2, a * grad1 + b * grad2
    return f1 + a * f2, grad1 + a * grad2


f, jac = scipinize(lambda x: obj_fun(x.reshape(-1, n_dim)))
x_init = points.flatten()

bounds = Bounds(lb = np.zeros(n_dim * n_point), ub = np.ones(n_dim * n_point))

slsqp_option = {
    "maxiter": 1000
}

res = minimize(
    f,
    x_init,
    method="SLSQP",
    jac=jac,
    bounds=bounds,
    options=slsqp_option,
)

points_sol = res.x.reshape(-1, n_dim)
print(res)

plt.scatter(points_sol[:, 0], points_sol[:, 1])
plt.show()
