import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(-2., 2.)
ax.set_ylim(-2., 2.)

class MovingCircle(object):
    def __init__(self):
        self.pos = np.zeros(2)
        patch = patches.Circle(xy=self.pos, radius=0.3, fc='r', ec='r')
        self.handle = ax.add_patch(patch)
        fig.canvas.mpl_connect('key_press_event', self.on_press)

    def update(self):
        vec = np.random.randn(2) * 0.04 # random warlking
        self.pos += np.array(vec)
        self.handle.set_center(xy=self.pos)
        ax.set_xlim(-2., 2.)
        ax.set_ylim(-2., 2.)

    def on_press(self, event):
        print('press', event.key)
        if event.key == 'left':
            self.pos[0] -= 0.1
        elif event.key == 'right':
            self.pos[0] += 0.1
        elif event.key == 'up':
            self.pos[1] += 0.1
        elif event.key == 'down':
            self.pos[1] -= 0.1
        sys.stdout.flush()

mc = MovingCircle()
while True:
    time.sleep(0.1)
    mc.update()
    fig.canvas.draw()
    fig.canvas.flush_events()
