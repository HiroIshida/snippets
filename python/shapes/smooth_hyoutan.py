import math
import numpy as np

a = 1.5
b = 1.0

def isInside(x):
    r_polar = np.linalg.norm(x)
    theta_polar = math.atan2(x[1], x[0])
    return r_polar < a + b * math.cos(theta_polar * 2.0)

if __name__=='__main__':
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    xlin = np.linspace(-4.0, 4.0, 100)
    ylin = np.linspace(-4.0, 4.0, 100)
    Xmesh, Ymesh = np.meshgrid(xlin, ylin)
    pts = np.array(zip(Xmesh.flatten(), Ymesh.flatten()))
    labels = np.array([isInside(pt) for pt in pts])

    ax.scatter(pts[labels, 0], pts[labels, 1], c="blue")
    plt.scatter(pts[~labels, 0], pts[~labels, 1], c="red")
    plt.show()

