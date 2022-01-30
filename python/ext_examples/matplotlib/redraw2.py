from os import truncate
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import numpy as np

class DetectionSimulator:
    def __init__(self, x=100., y=100., w=60., h=60.):
        self.true_state = np.array([x, y, w, h])
        self.true_velocity = np.zeros(4)
        self.phase = 0 # 0 : normal, 1: anomarry

        self.est_state = self.true_state
        self.true_bbox_patch = None

        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ax.set_xlim(0, 300)
        ax.set_ylim(0, 300)
        self.fax = (fig, ax)

    def update(self, dt):
        self.true_state += self.true_velocity * dt
        self.true_state[0] = min(max(self.true_state[0], 0), 300)
        self.true_state[1] = min(max(self.true_state[1], 0), 300)
        self.true_state[2] = min(max(self.true_state[2], 20), 150)
        self.true_state[3] = min(max(self.true_state[3], 20), 150)
        self.true_velocity = np.random.randn(4) * 40
        self.est_state = self.true_state + np.random.randn(4)

    def redraw(self):
        fig, ax = self.fax
        if self.true_bbox_patch is None:
            self.true_bbox_patch = patches.Rectangle(
                    self.true_state[:2], 
                    self.true_state[2], 
                    self.true_state[3],
                    fill = False,
                    )
            ax.add_patch(self.true_bbox_patch)
        else:
            self.true_bbox_patch.set_xy(self.true_state[:2])
            self.true_bbox_patch.set_width(self.true_state[2])
            self.true_bbox_patch.set_height(self.true_state[3])
            fig.canvas.draw()

        print(self.true_state)
if __name__=='__main__':
    sim = DetectionSimulator()
    for i in range(300):
        sim.update(0.1)
        sim.redraw()
