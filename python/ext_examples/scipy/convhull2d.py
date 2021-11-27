from scipy.spatial import ConvexHull
import numpy as np

def linear_constraint_from_polygon(P_transpose):
    P = P_transpose.T

    normalize = lambda x: x/np.linalg.norm(x)

    # create tmp frame:
    origin = P[1]
    e1: np.ndarray = normalize(P[1] - P[0])
    e2_tmp: np.ndarray = P[2] - P[1]
    assert abs(e2_tmp.dot(e1)) > 1e-4, "must be linearly indep"

    e3 = normalize(np.cross(e1, e2_tmp))
    e2 = np.cross(e3, e1)
    M = np.vstack([e1, e2, e3]).T
    M_convert = np.array([[1, 0, 0], [0, 1, 0]]).dot(M.T)
    P_projected = M_convert.dot((P.T - origin).T)
    chull = ConvexHull(P_projected.T)

    n_dim_hull = 3 - 1
    A_hull = chull.equations[:, :n_dim_hull] # 3x 2
    A = A_hull.dot(M_convert)
    b = -chull.equations[:, n_dim_hull] - origin.dot(A.T)
    return A, b

if __name__=='__main__':
    P = np.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.]])
    A, b = linear_constraint_from_polygon(P)

    Q = np.random.rand(100000, 3) * 5 - np.ones(3) * 2
    values = A.dot(Q.T).T - b
    isinside = np.all(values < 0.0, axis=1)
    Qin = Q[isinside]

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(211 , projection='3d')
    ax.scatter(Qin[:, 0], Qin[:, 1], Qin[:, 2], s=1)
