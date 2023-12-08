from sdf.d2 import polygon, extrude
from sdf.d3 import cylinder, capped_cylinder, X, Y, Z
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
dilated = ls.buffer(0.0025)
x, y = dilated.exterior.xy
pts = np.vstack([x, y]).T

f_handle2d = polygon(pts[1:])
f_handle3d = extrude(f_handle2d, 0.018).rotate(np.pi * 0.5, X).translate(Z * 0.0475).translate(X * 0.048)
f_body = capped_cylinder(np.zeros(3), Z * 0.095, 0.04)
f_body_sub = capped_cylinder(Z * 0.005, Z * 0.1, 0.036)
f = (f_body - f_body_sub) | f_handle3d
out = "hoge.stl"
f.save(out, samples=50000)
print("NOTE: please use meshalb to reduce the number of triangles")
print("NOTE: recommendation is using marching cube + quadric edge collapse decimation")
