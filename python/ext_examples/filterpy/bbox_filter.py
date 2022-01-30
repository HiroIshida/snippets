import copy
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from typing import Optional
import numpy as np

class BoxDrawer:
    patch: Optional[patches.Rectangle]
    color: str
    def __init__(self, fax, color='black'):
        self.color = color
        self.fax = fax
        self.patch = None

    def draw(self, x, y, w, h):
        if self.patch is None:
            self.patch = patches.Rectangle((x, y), w, h, ec=self.color, fill=False)
            fig, ax = self.fax
            ax.add_patch(self.patch)
        else:
            self.patch.set_xy((x, y))
            self.patch.set_width(w)
            self.patch.set_height(h)

class BboxKalmanFilter(KalmanFilter):
    def __init__(self, x):
        super().__init__(dim_x=4, dim_z=4)
        self.x = x
        self.F = np.eye(4)
        self.Q = np.eye(4) * 30 ** 2

        self.H = np.eye(4)
        self.R = np.eye(4) * 60 ** 2

    def update(self, sensor_value, dt):
        self.predict()
        super().update(sensor_value)
        return self.x

class DetectionSimulator:
    def __init__(self, x=50., y=50., w=60., h=60.,
            b_min = np.array([0, 0]),
            b_max = np.array([300, 300]),
            enable_anomary=True
            ):
        self.true_state = np.array([x, y, w, h])
        self.true_velocity = np.zeros(4)
        self.phase = 0 # 0 : normal, 1: anomarry

        self.est_state = self.true_state
        self.filter = BboxKalmanFilter(self.true_state)
        self.enable_anomary = enable_anomary

        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ax.set_xlim(b_min[0], b_max[0])
        ax.set_ylim(b_min[1], b_max[1])

        self.b_min = b_min
        self.b_max = b_max
        self.fax = (fig, ax)
        self.true_box_drawer = BoxDrawer(self.fax)
        self.est_box_drawer = BoxDrawer(self.fax, color='red')
        self.kf_box_drawer = BoxDrawer(self.fax, color='green')

    def state_cropper(self, state):
        state_new = copy.deepcopy(state)
        state_new[0] = min(max(state[0], self.b_min[0]), self.b_max[0])
        state_new[1] = min(max(state[1], self.b_min[1]), self.b_max[1])
        state_new[2] = min(max(state[2], 20), 150)
        state_new[3] = min(max(state[3], 20), 150)
        return state_new

    def update(self, dt):
        if np.random.rand() < 0.02 and self.enable_anomary:
            self.phase = (self.phase + 1)%2
        self.true_state = self.state_cropper(self.true_state + self.true_velocity * dt)
        self.true_velocity = np.random.randn(4) * 20

        dummy_base = np.array([150, 150, 100, 100])
        est_base = self.true_state if self.phase==0 else dummy_base
        self.est_state = self.state_cropper(est_base + np.random.randn(4) * 10)
        self.kf_state = self.filter.update(self.est_state, dt)

    def redraw(self):
        fig, ax = self.fax
        self.true_box_drawer.draw(*self.true_state)
        self.est_box_drawer.draw(*self.est_state)
        self.kf_box_drawer.draw(*self.kf_state)
        fig.canvas.draw()

if __name__=='__main__':
    sim = DetectionSimulator()
    for i in range(300):
        sim.update(0.1)
        sim.redraw()
