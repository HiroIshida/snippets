import time
import numpy as np
from skrobot.coordinates import CascadedCoords, make_cascoords
from skrobot.model import Axis
from skrobot.viewers import TrimeshSceneViewer

co_parent = make_cascoords(pos = [0, 0, 1.0])
co_child = CascadedCoords(parent=co_parent, pos = [1.0, 0.0, 0.0])
ax_parent = Axis.from_cascoords(co_parent)
ax_child = Axis.from_cascoords(co_child)

vis = TrimeshSceneViewer()
vis.add(ax_parent)
vis.add(ax_child)
vis.show()

for _ in range(1000):
    co_parent.rotate(0.1, "z")
    vis.redraw()
    time.sleep(0.2)
