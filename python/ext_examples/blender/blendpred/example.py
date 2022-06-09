import bpy
import sys
import math
import os
import pathlib

working_dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(working_dir_path)

# force append
sys.path.extend(['/opt/ros/galactic/lib/python3.8/site-packages', '/home/h-ishida/.pyenv/versions/3.10.2/lib/python3.10', '/home/h-ishida/.pyenv/versions/3.10.2/lib/python3.10/lib-dynload', '/home/h-ishida/.pyenv/versions/3.10.2/lib/python3.10/site-packages', '/home/h-ishida/documents/python/blendpred'])

import blendpred.utils as utils
from blendpred.mesh import create_mesh_from_file


# Args
output_file_path = bpy.path.relpath(str(sys.argv[sys.argv.index('--') + 1]))
resolution_percentage = int(sys.argv[sys.argv.index('--') + 2])
num_samples = int(sys.argv[sys.argv.index('--') + 3])

# Scene Building

## Reset
utils.clean_objects()

## Camera

#utils.add_track_to_constraint(camera_object, center_suzanne)
#utils.set_camera_params(camera_object.data, center_suzanne, lens=50.0)

## Lights

# Render Setting
scene = bpy.data.scenes["Scene"]
file_path = pathlib.Path("/opt/ros/noetic/share/pr2_description/meshes/upper_arm_v0/forearm_roll_L.stl")
obj = create_mesh_from_file(scene, file_path, "hoge", "hoge")

camera_object = utils.create_camera(location=(10.0, -7.0, 0.0))

utils.add_track_to_constraint(camera_object, obj)
utils.set_camera_params(camera_object.data, obj, lens=50.0)

utils.create_sun_light(rotation=(0.0, math.pi * 0.5, -math.pi * 0.1))

utils.set_output_properties(scene, resolution_percentage, output_file_path)
utils.set_cycles_renderer(scene, camera_object, num_samples)
