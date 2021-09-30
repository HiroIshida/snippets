import scipy.interpolate  
import numpy as np
import matplotlib.pyplot as plt 

xlin = np.linspace(0.0, 2.0, 200)
ylin = np.linspace(0.0, 1.0, 200)
X, Y = np.meshgrid(xlin, ylin)

Z = X**2 + Y**2

fig, ax = plt.subplots()
X, Y = np.meshgrid(xlin, ylin)
ax.contourf(X, Y, Z)
plt.show()
