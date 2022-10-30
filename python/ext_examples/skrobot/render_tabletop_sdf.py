from dataclasses import dataclass
import plotly
import numpy as np
from typing import List, Tuple
from skrobot.model.link import Link
from skrobot.model.primitives import Box, Axis, Cylinder
from skrobot.coordinates import CascadedCoords
from skrobot.sdf import UnionSDF


@dataclass
class SDFMesh:
    dimension: np.ndarray
    mesh: np.ndarray

    def get_mgrid(self) -> Tuple[np.ndarray, ...]:
        self.mesh.shape
        xlin = np.linspace(0.0, self.dimension[0], self.mesh.shape[0])
        ylin = np.linspace(0.0, self.dimension[1], self.mesh.shape[1])
        zlin = np.linspace(0.0, self.dimension[2], self.mesh.shape[2])
        X, Y, Z = np.meshgrid(xlin, ylin, zlin)
        return X, Y, Z

    def visualize(self, contour: bool = False):
        X, Y, Z = self.get_mgrid()
        if contour:
            fig = go.Figure(data=go.Isosurface(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=self.mesh.flatten(),
                isomin=-0.02,
                isomax=0.02,
                opacity=0.1,
                surface_count=3,
                colorbar_nticks=5,
                colorscale='Plotly3',
                caps=dict(x_show=False, y_show=False)
                ))
        else:
            fig = go.Figure(data=go.Volume(
                x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                value=mesh.mesh.flatten(),
                isomin=-0.1,
                isomax=0.3,
                opacity=0.1,
                surface_count=25,
                colorscale='Plotly3'
                ))
        fig.show()


@dataclass
class TableTopWorld:
    table: Box
    obstacles: List[Link]

    def get_sdfmesh(self, grid_sizes: Tuple[int, int, int] = (56, 56, 28), mesh_height: float = 0.3) -> SDFMesh:
        depth, width, height = self.table._extents
        xlin = np.linspace(-0.5 * depth, 0.5 * depth, grid_sizes[0])
        ylin = np.linspace(-0.5 * width, 0.5 * width, grid_sizes[1])
        zlin = np.linspace(0.5 * height, 0.5 * height + mesh_height, grid_sizes[2])
        X, Y, Z = np.meshgrid(xlin, ylin, zlin)
        pts_local = np.array(list(zip(X.flatten(), Y.flatten(), Z.flatten())))
        tf = self.table.get_transform()
        pts_world = tf.transform_vector(pts_local)

        sdf = UnionSDF([obs.sdf for obs in self.obstacles])
        values = sdf.__call__(pts_world)
        mesh = values.reshape(grid_sizes)
        dimension = [depth, width, mesh_height]
        return SDFMesh(dimension, mesh)


def create_tabletop_world() -> TableTopWorld:
    # 73b2 table
    table_depth = 0.5
    table_width = 0.75
    table_height = 0.7
    table = Box(extents=[table_depth, table_width, table_height], with_sdf=True)
    x = 0.4 + table_depth * 0.5 + np.random.rand() * 0.2
    y = -0.2 + np.random.rand() * 0.4
    z = 0
    table.translate([x, y, z])

    table_tip = table.copy_worldcoords()
    table_tip.translate([-table_depth * 0.5, -table_width * 0.5, +0.5 * table_height])

    n_box = np.random.randint(3)
    n_cylinder = np.random.randint(8)

    obstacles = []

    for _ in range(n_box):
        dimension = np.array([0.1, 0.1, 0.05]) + np.random.rand(3) * np.array([0.2, 0.2, 0.2])
        box = Box(extents=dimension, with_sdf=True)

        co = table_tip.copy_worldcoords()
        box.newcoords(co)
        x = dimension[0] * 0.5 + np.random.rand() * (table_depth - dimension[0])
        y = dimension[1] * 0.5 + np.random.rand() * (table_width - dimension[1])
        z = dimension[2] * 0.5
        box.translate([x, y, z])
        obstacles.append(box)

    for _ in range(n_cylinder):
        r = np.random.rand() * 0.03 + 0.01
        h = np.random.rand() * 0.2 + 0.05
        cylinder = Cylinder(radius=r, height=h, with_sdf=True)

        co = table_tip.copy_worldcoords()
        cylinder.newcoords(co)
        x = r + np.random.rand() * (table_depth - r)
        y = r + np.random.rand() * (table_width - r)
        z = 0.5 * h
        cylinder.translate([x, y, z])
        obstacles.append(cylinder)

    return TableTopWorld(table, obstacles)


if __name__ == "__main__":
    import skrobot
    import time
    import plotly.graph_objects as go
    viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
    world = create_tabletop_world()
    mesh = world.get_sdfmesh()
    mesh.visualize()

    viewer.add(world.table)
    for obs in world.obstacles:
        viewer.add(obs)

    viewer.show()

    print("==> Press [q] to close window")
    while not viewer.has_exit:
        time.sleep(0.1)
        viewer.redraw()
