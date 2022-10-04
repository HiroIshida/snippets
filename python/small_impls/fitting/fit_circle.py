from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import minimize
import time


@dataclass
class Circle:
    center: np.ndarray
    radius: float


def find_circle(X: np.ndarray) -> Circle:
    assert X.ndim == 2
    assert X.shape[1] == 2

    def fun(arg):
        c1, c2, r = arg
        C = np.array([c1, c2])
        diffs = np.sqrt(np.sum((X - C)**2, axis=1)) - r
        cost = np.sum(diffs ** 2)
        return cost

    pcloud_center = np.mean(X, axis=0)
    r = 0.1
    x0 = np.hstack([pcloud_center, r])
    ts = time.time()
    res = minimize(fun, x0=x0, method='BFGS')
    return Circle(res.x[:2], res.x[2])


center_true = np.ones(2) * 0.3
r_center = 0.4
points = []
for i in range(1000):
    angle = np.random.randn()
    point = center_true + (r_center + np.random.randn() * 0.05) * np.array([np.cos(angle), np.sin(angle)])
    points.append(point)

X = np.array(points)

fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1])
circle = find_circle(X[:, :2])
plt_circle = plt.Circle(circle.center, circle.radius, color='r', fill=False)
ax.add_patch(plt_circle)
plt.xlabel("x-label")
plt.ylabel("y-label")
plt.show()
