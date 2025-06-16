#!/usr/bin/env python3
"""
Minimal Sapien demo:
Spawns a fixed-base Franka Panda, a table, a red cube, and renders the scene.
No Gym, no RL – just pure Sapien.
"""

import numpy as np
import sapien
from sapien import Pose
from sapien.utils.viewer import Viewer


TABLE_HEIGHT = 0.8          # Height of the tabletop (m)
TIMESTEP = 1.0 / 100.0      # Physics-engine timestep (s)
INIT_QPOS = [               # Eight joints + two-finger gripper = 9 DoF
    0.0, 0.196349541, 0.0, -2.61799388, 0.0,
    2.94159265, 0.78539816, 0.0, 0.0
]


def build_world(scene):
    """Create ground, table, cube and Panda robot."""
    # ---- Ground -------------------------------------------------------------
    material = scene.create_physical_material(static_friction=1.0,
                                              dynamic_friction=1.0,
                                              restitution=0.0)
    scene.default_physical_material = material
    scene.add_ground(0.0)

    # ---- Table (kinematic) --------------------------------------------------
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[0.4, 0.4, 0.025])
    builder.add_box_visual(half_size=[0.4, 0.4, 0.025])
    table = builder.build_kinematic(name="table")
    table.set_pose(Pose([0.0, 0.0, TABLE_HEIGHT - 0.025]))

    # ---- Red cube (dynamic) -------------------------------------------------
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[0.02, 0.02, 0.02])
    builder.add_box_visual(half_size=[0.02, 0.02, 0.02],
                           material=[1.0, 0.0, 0.0])  # RGB
    cube = builder.build(name="cube")
    cube.set_pose(Pose([0.0, 0.0, TABLE_HEIGHT + 0.02]))

    # ---- Panda robot --------------------------------------------------------
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True               # Base will not move
    robot = loader.load("panda/panda.urdf")
    robot.set_name("panda")
    robot.set_root_pose(Pose([-0.56, 0.0, TABLE_HEIGHT]))
    robot.set_qpos(INIT_QPOS)

    # Disable gravity for links (optional – matches original env behaviour)
    for link in robot.get_links():
        link.disable_gravity = True

    return robot, cube


def setup_viewer(scene):
    """Create a viewer, add lights and set an initial camera pose."""
    # Lighting
    scene.set_ambient_light([0.4, 0.4, 0.4])
    scene.add_directional_light([1, -1, -1], [0.3, 0.3, 0.3])
    scene.add_directional_light([0, 0, -1], [1.0, 1.0, 1.0])

    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.set_camera_xyz(x=1.5, y=0.0, z=2.0)
    viewer.set_camera_rpy(y=3.14, p=-0.5, r=0.0)
    return viewer


def main():
    scene = sapien.Scene()  # Create an instance of simulation world (aka scene)
    scene.set_timestep(1 / 100.0)  # Set the simulation frequency
    build_world(scene)
    viewer = setup_viewer(scene)
    while not viewer.closed:
        scene.step()
        scene.update_render()
        viewer.render()



if __name__ == "__main__":
    main()
