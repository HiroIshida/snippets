import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# see
# https://stackoverflow.com/questions/4098131/how-to-update-a-plot-in-matplotlib

plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

c = patches.Circle(xy=(0, 0), radius=1, fc='r', ec='r')
patch = ax.add_patch(c)

c2 = patches.Circle(xy=(0, 0), radius=1, fc='r', ec='r')
patch2 = ax.add_patch(c2)

for i in range(100):
    patch2.set_center((0, 0.1*i))
    fig.canvas.draw()
    fig.canvas.flush_events()
