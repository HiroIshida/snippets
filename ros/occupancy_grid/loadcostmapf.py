import scipy.interpolate  
import copy
import numpy as np
import matplotlib.pyplot as plt 
import pickle 

with open("costmapf.pickle", "rb") as f:
    costmapdata = pickle.load(f)

costmapf = costmapdata.convert2sdf()

b = 4.0
xlin = np.linspace(-b, b, 200)
ylin = np.linspace(-b, b, 200)
X, Y = np.meshgrid(xlin, ylin)
pts = np.array(list(zip(X.flatten(), Y.flatten())))
Z_ = costmapf(pts)
Z = Z_.reshape((200, 200))
fig, ax = plt.subplots()
c = ax.contourf(X, Y, Z)
cbar = fig.colorbar(c)

idxes_clear = Z_ < 80
pts_valid = pts[idxes_clear, :]
ax.scatter(pts_valid[:, 0], pts_valid[:, 1])
plt.show()
