import pickle 
import scipy.interpolate  
import numpy as np
import matplotlib.pyplot as plt 
import tf

class MapData:
    def __init__(self, arr, res, origin, tf_base_to_map):
        self.arr = arr
        self.tf_base_to_map = tf_base_to_map
        self.origin = origin
        self.res = res

with open("localcost_pr2.pickle", 'rb') as f:
    data = pickle.load(f)

nx, ny = data.arr.shape
bmin = [0, 0]
bmax = [nx * data.res, ny * data.res]
xlin = np.linspace(bmin[0], bmax[0], 200)
ylin = np.linspace(bmin[1], bmax[1], 200)
f = scipy.interpolate.interp2d(xlin, ylin, np.fliplr(data.arr).T, kind='cubic')
fp = scipy.interpolate.RegularGridInterpolator((xlin, ylin), np.fliplr(data.arr), method='linear')

fig, ax = plt.subplots()
X, Y = np.meshgrid(xlin, ylin)
Z = f(xlin, ylin)
ax.contourf(X, Y, Z)

pos, rot = data.tf_base_to_map
pos_origin = np.array([0.0, 0.0])

M = tf.transformations.quaternion_matrix(rot)[:2, :2]
points = pos_origin.dot(M) + np.array([pos[0], pos[1]])
mappos_origin = np.array([data.origin.position.x, data.origin.position.y])
points = points - mappos_origin

#fs = fp(points)
#idxes = fs < 30
#ax.scatter(points[idxes, 0], points[idxes, 1]) 

npots = points.shape[0]
print(points)
ax.scatter(points[0], points[1]) 



plt.show()






