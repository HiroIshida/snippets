import pickle 
import scipy.interpolate  
import numpy as np
import matplotlib.pyplot as plt 
import tf

class MapData:
    def __init__(self, arr, res, origin, tf_base_to_odom):
        self.arr = arr
        self.tf_base_to_odom = tf_base_to_odom
        self.origin = origin
        self.res = res

with open("localcost_pr2.pickle", 'rb') as f:
    data = pickle.load(f)

nx, ny = data.arr.shape
bmin = [0, 0]
bmax = [nx * data.res, ny * data.res]
xlin = np.linspace(bmin[0], bmax[0], 200)
ylin = np.linspace(bmin[1], bmax[1], 200)
finterp = scipy.interpolate.interp2d(xlin, ylin, np.flip(data.arr.T, axis=0), kind='cubic')
fp = scipy.interpolate.RegularGridInterpolator((xlin, ylin), np.flip(data.arr.T, axis=0), method='linear')

fig, ax = plt.subplots()
X, Y = np.meshgrid(xlin, ylin)
Z = finterp(xlin, ylin)
ax.contourf(X, Y, Z.T)

def base_to_map(P): # P|base -> P|map
    n_points = len(P)
    pos, rot = data.tf_base_to_odom
    M = tf.transformations.quaternion_matrix(rot)[:2, :2]
    points = P.dot(M.T) + np.repeat(np.array([[pos[0], pos[1]]]), n_points, 0)

    mappos_origin = np.array([[data.origin.position.x, data.origin.position.y]])
    points = points - np.repeat(mappos_origin, n_points, 0)
    return points

points = np.array(zip(X.flatten(), Y.flatten()))
fs = fp(points)
idxes = fs < 30
ax.scatter(points[idxes, 0], points[idxes, 1]) 

P_origin = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
P_map = base_to_map(P_origin)
ax.scatter(P_map[0, 0], P_map[0, 1], c="black", s=30) 
ax.scatter(P_map[1, 0], P_map[1, 1], c="red", s=30) 
ax.scatter(P_map[2, 0], P_map[2, 1], c="green", s=30) 
plt.show()






