import scipy.interpolate  
import copy
import numpy as np
import matplotlib.pyplot as plt 
import dill 

with open("costmapf.dill", "rb") as f:
    costmapf = dill.load(f)

b = 1.0
xlin = np.linspace(-b, b, 200)
ylin = np.linspace(-b, b, 200)
X, Y = np.meshgrid(xlin, ylin)
pts = np.array(list(zip(X.flatten(), Y.flatten())))
Z_ = costmapf(pts)
Z = Z_.reshape((200, 200))
fig, ax = plt.subplots()
ax.contourf(X, Y, Z)
plt.show()
