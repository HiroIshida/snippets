from shapely.geometry import LineString
import numpy as np
import matplotlib.pyplot as plt


def ellipse_point(a, b, theta):
    r = a * b / np.sqrt(a**2 * np.sin(theta)**2 + b**2 * np.cos(theta)**2)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.array([x, y])


a = 0.025
b = 0.0325
points = np.array([ellipse_point(a, b, theta) for theta in np.linspace(-np.pi * 0.6, np.pi * 0.6, 20)])
ls = LineString(points)
dilated = ls.buffer(0.005)

# visualize
x, y = dilated.exterior.xy
plt.figure()
plt.plot(points[:, 0], points[:, 1])
plt.plot(x, y)
plt.fill(x, y, alpha=0.3)
plt.axis('equal')
plt.show()
