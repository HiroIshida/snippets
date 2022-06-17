# blender --background --python 02_suzanne.py --render-frame 1 -- </path/to/output/image> <resolution_percentage> <num_samples>

import bpy
from mathutils import Vector, Quaternion
import sys
import math
import os
import pickle

working_dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(working_dir_path)

import utils
import numpy as np


def set_scene_objects() -> bpy.types.Object:
    num_suzannes = 15
    for index in range(num_suzannes):
        utils.create_smooth_monkey(location=((index - (num_suzannes - 1) / 2) * 3.0, 0.0, 0.0),
                                   name="Suzanne" + str(index))
    return bpy.data.objects["Suzanne" + str(int((num_suzannes - 1) / 2))]


# Args
output_file_path = bpy.path.relpath(str(sys.argv[sys.argv.index('--') + 1]))
resolution_percentage = 30
num_samples = int(sys.argv[sys.argv.index('--') + 3])

# Scene Building

## Reset
utils.clean_objects()

## Suzannes
center_suzanne = set_scene_objects()

## Camera
cam = utils.create_camera(location=(10.0, -7.0, 0.0))
utils.add_track_to_constraint(cam, center_suzanne)
utils.set_camera_params(cam.data, center_suzanne, lens=50.0)

bpy.context.scene.render.resolution_percentage = resolution_percentage
frame = cam.data.view_frame(scene=bpy.context.scene)
topRight = frame[0]
bottomRight = frame[1]
bottomLeft = frame[2]
topLeft = frame[3]
resolutionX = int(bpy.context.scene.render.resolution_x * (bpy.context.scene.render.resolution_percentage / 100))
resolutionY = int(bpy.context.scene.render.resolution_y * (bpy.context.scene.render.resolution_percentage / 100))

# setup vectors to match pixels
xRange = np.linspace(topLeft[0], topRight[0], resolutionX)
yRange = np.linspace(topLeft[1], bottomLeft[1], resolutionY)
values = np.empty((xRange.size, yRange.size), dtype=object)

# indices for array mapping
indexX = 0
indexY = 0

# filling array with None
for x in xRange:
    for y in yRange:
        values[indexX,indexY] = None
        indexY += 1
    indexX += 1
    indexY = 0

targets = [bpy.data.objects["Suzanne" + str(i)] for i in range(15)]

points = []
# iterate over all targets
for target in targets:
    print(target.name)
    # calculate origin
    matrixWorld = target.matrix_world
    matrixWorldInverted = matrixWorld.inverted()
    origin = matrixWorldInverted @ cam.matrix_world.translation
            
    # reset indices
    indexX = 0
    indexY = 0
    
    # iterate over all X/Y coordinates
    for x in xRange:
        for y in yRange:
            # get current pixel vector from camera center to pixel
            pixelVector = Vector((x, y, topLeft[2]))
            
            # rotate that vector according to camera rotation
            pixelVector.rotate(cam.matrix_world.to_quaternion())

            # calculate direction vector
            destination = matrixWorldInverted @ (pixelVector + cam.matrix_world.translation) 
            direction = (destination - origin).normalized()
            
            # perform the actual ray casting
            hit, location, norm, face =  target.ray_cast(origin, direction)
            
            if hit:
                pt: Vector = (matrixWorld @ location)
                points.append(np.array([pt.x, pt.y, pt.z]))
            
            # update indices
            indexY += 1
                        
        indexX += 1
        indexY = 0

with open("cloud.pkl", "wb") as f:
    pickle.dump(np.array(points), f)

## Lights
utils.create_sun_light(rotation=(0.0, math.pi * 0.5, -math.pi * 0.1))

# Render Setting
scene = bpy.data.scenes["Scene"]
utils.set_output_properties(scene, resolution_percentage, output_file_path)
utils.set_cycles_renderer(scene, cam, num_samples)
