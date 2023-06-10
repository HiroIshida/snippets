import math
import numpy as np
from dataclasses import dataclass
from typing import List
import matplotlib.pyplot as plt
import matplotlib.patches as patches



@dataclass
class PlanerCoords:
    pos: np.ndarray
    angle: float

    @classmethod
    def create(cls, x, y, angle) -> "PlanerCoords":
        return PlanerCoords(np.array([x, y]), angle)

    @classmethod
    def standard(cls) -> "PlanerCoords":
        return PlanerCoords.create(0.0, 0.0, 0.0)


def rotation_matrix_2d(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    rotmat = np.array([[c, s], [-s, c]])
    return rotmat


@dataclass
class Box2d:
    extent: np.ndarray
    coords: PlanerCoords

    def get_verts(self) -> np.ndarray:
        half_extent = self.extent * 0.5
        dir1 = rotation_matrix_2d(self.coords.angle).dot(half_extent)

        half_extent_rev = half_extent
        half_extent_rev[0] *= -1.0

        dir2 = rotation_matrix_2d(self.coords.angle).dot(half_extent_rev)

        v1 = self.coords.pos + dir1
        v2 = self.coords.pos + dir2
        v3 = self.coords.pos - dir1
        v4 = self.coords.pos - dir2
        return np.array([v1, v2, v3, v4])

    def visualize(self, fax, color="red") -> None:
        fig, ax = fax
        verts = self.get_verts()
        ax.scatter(verts[:, 0], verts[:, 1], c=color)

        idx_pairs = [[0, 1], [1, 2], [2, 3], [3, 0]]
        for i, j in idx_pairs:
            v1 = verts[i]
            v2 = verts[j]
            ax.plot([v1[0], v2[0]], [v1[1], v2[1]], color=color)

    def sd(self, points):
        n_pts, _ = points.shape
        half_extent = self.extent * 0.5

        pts_from_center = points - self.coords.pos
        sd_vals_each_axis = np.abs(pts_from_center) - half_extent[None, :]

        positive_dists_each_axis = np.maximum(sd_vals_each_axis, 0.0)
        positive_dists = np.sqrt(np.sum(positive_dists_each_axis**2, axis=1))

        negative_dists_each_axis = np.max(sd_vals_each_axis, axis=1)
        negative_dists = np.minimum(negative_dists_each_axis, 0.0)

        sd_vals = positive_dists + negative_dists
        return sd_vals

    def is_colliding(self, box: "Box2d") -> bool:
        vals = self.sd(box.get_verts())
        if np.any(vals < 0.0):
            return True
        vals = box.sd(self.get_verts())
        return bool(np.any(vals < 0.0))


def sample_box(table_extent: np.ndarray, box_extent: np.ndarray, obstacles: List[Box2d]) -> Box2d:
    table = Box2d(table_extent, PlanerCoords.standard())

    while True:
        box_pos_cand = -0.5 * table_extent + table_extent * np.random.rand(2)
        angle_cand = np.random.rand() * np.pi
        box_cand = Box2d(box_extent, PlanerCoords(box_pos_cand, angle_cand))

        def is_valid(box_cand):
            is_inside = np.all(table.sd(box_cand.get_verts()) < 0.0)
            if is_inside:
                for obs in obstacles:
                    if box_cand.is_colliding(obs):
                        return False
                return True
            return False

        if is_valid(box_cand):
            return box_cand


# np.random.seed(1)

table_extent = np.array([0.8, 0.5])
table = Box2d(table_extent, PlanerCoords.standard())
box2d = sample_box(table_extent, np.array([0.2, 0.2]), [])

obstacles = [box2d]
for _ in range(4):
    obs = sample_box(table_extent, np.array([0.1, 0.1]), obstacles)
    obstacles.append(obs)

fig, ax = plt.subplots()
table.visualize((fig, ax), "red")
box2d.visualize((fig, ax), "blue")
for obs in obstacles[1:]:
    obs.visualize((fig, ax), "green")

ax.set_xlim([-1.0, 1.0])
ax.set_ylim([-1.0, 1.0])
plt.show()
