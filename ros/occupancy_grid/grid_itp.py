import pickle 
import scipy.interpolate  
import numpy as np
import matplotlib.pyplot as plt 

with open("localcost_pr2.pickle", 'rb') as f:
    arr = pickle.load(f)

bmin = [-5, -5]
bmax = [5, 5]
xlin = np.linspace(bmin[0], bmax[0], 200)
ylin = np.linspace(bmin[1], bmax[1], 200)
f = scipy.interpolate.interp2d(xlin, ylin, np.fliplr(arr).T, kind='cubic')
fp = scipy.interpolate.RegularGridInterpolator((xlin, ylin), np.fliplr(arr), method='linear')


fig, ax = plt.subplots()
X, Y = np.meshgrid(xlin, ylin)
data = f(xlin, ylin)
ax.contourf(X, Y, data)

points = np.array(zip(X.flatten(), Y.flatten()))
fs = fp(points)
idxes = fs < 30
ax.scatter(points[idxes, 0], points[idxes, 1]) 
plt.show()

