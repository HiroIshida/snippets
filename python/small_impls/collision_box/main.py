from dataclasses import dataclass
from typing import List
import numpy as np
import math
from math import sin, cos

def cross_2d(a: np.ndarray, b: np.ndarray) -> float:
    return a[0] * b[1] - b[0] * a[1]

@dataclass
class Edge:
    p0: np.ndarray
    p1: np.ndarray

class CollisionBox:
    points: List[np.ndarray]
    visibl_edges: List[Edge]

    def __init__(self, x, y, yaw, length, width):
        points = []
        points.append(np.array([+0.5 * length, +0.5 * width]))
        points.append(np.array([-0.5 * length, +0.5 * width]))
        points.append(np.array([-0.5 * length, -0.5 * width]))
        points.append(np.array([+0.5 * length, -0.5 * width]))

        rot = np.array([[cos(yaw), -sin(yaw)], [sin(yaw), cos(yaw)]])

        for i in range(4):
            points[i] = rot.dot(points[i]) + np.array([x, y])
        
        visible_edges = []
        for i in range(4):
            i_next = 0 if i ==3 else i + 1
            p0 = points[i]
            p1 = points[i_next]

            is_visible = cross_2d(p1 - p0, p0) > 0.0
            if is_visible:
                visible_edges.append(Edge(p0, p1))

        self.points = points
        self.visibl_edges = visible_edges 

    def distance(self, angle):
        assert angle < math.pi and angle > -math.pi
        v = np.array([cos(angle), sin(angle)])
        dist = np.inf
        for edge in self.visibl_edges:
            w = edge.p1 - edge.p0
            s, t = np.linalg.inv(np.stack([v, w]).T).dot(edge.p1)
            is_hit = (0.0 < t and t < 1.0) and s > 0.0
            if is_hit:
                dist = s
        return dist

    def visualize(self, fax):
        fig, ax = fax
        for i in range(4):
            i_next = 0 if i ==3 else i + 1
            p0 = self.points[i]
            p1 = self.points[i_next]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'green')


@dataclass
class CollisionBoxes:
    boxes: List[CollisionBox]

    def distance(self, angle):
        dist_min = np.inf
        i_min = None
        for i in range(len(self.boxes)):
            box = self.boxes[i]
            dist = box.distance(angle)
            if dist == np.inf:
                continue
            if dist < dist_min:
                dist_min = dist
                i_min = i
        return dist_min


if __name__ == '__main__':
    box1 = CollisionBox(2., 2., 0., 2., 2.)
    box2 = CollisionBox(-1., 6., -0.3, 4., 8.)
    box3 = CollisionBox(-5., 2., +0.4, 2., 12.)
    boxes = CollisionBoxes([box1, box2, box3])

    boxes.distance(math.pi * 0.75)

    n_div = 360
    step = 2 * math.pi / n_div
    angle_zero = -math.pi + 1e-8
    points = []
    inf_points = []
    for i in range(n_div):
        angle = angle_zero + i * step
        dist = boxes.distance(angle)
        if math.isinf(dist):
            dist = 20.0
            inf_points.append(np.array([cos(angle) * dist, sin(angle) * dist]))
        else:
            points.append(np.array([cos(angle) * dist, sin(angle) * dist]))
    P_inf = np.array(inf_points)
    P = np.array(points)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    if len(P) > 0:
        for p in P:
            ax.plot([0, p[0]], [0, p[1]], color='r')
        ax.scatter(P[:, 0], P[:, 1], s=3)

    if len(P_inf) > 0:
        for p in P_inf:
            ax.plot([0, p[0]], [0, p[1]], color='b')
        ax.scatter(P_inf[:, 0], P_inf[:, 1], s=3)

    for box in boxes.boxes:
        box.visualize((fig, ax))
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.grid()
    plt.axis('equal')
    plt.show()



