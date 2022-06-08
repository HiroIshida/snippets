# Copyright 2022 The Kubric Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Hirokazu Ishida modified

import logging
import numpy as np
import kubric as kb
from pathlib import Path
from kubric.renderer.blender import Blender as KubricRenderer
from kubric.simulator.pybullet import PyBullet as KubricSimulator

logging.basicConfig(level="INFO")

# --- create scene and attach a renderer to it
scene = kb.Scene(resolution=(1024, 1024))
renderer = KubricRenderer(scene)
simulator = KubricSimulator(scene)

json_path = Path("~/kubric/gso_dataset/GSO.json").expanduser()
asset_source = kb.AssetSource.from_manifest(json_path)
asset_source.data_dir = Path("~/kubric/gso_dataset/GSO").expanduser()
obj = asset_source.create(asset_id="Marvel_Avengers_Titan_Hero_Series_Doctor_Doom")
scene += obj

# --- populate the scene with objects, lights, cameras
#scene += kb.Cube(name="floor", scale=(10, 10, 0.1), position=(0, 0, -0.1)) #scene += kb.Sphere(name="ball", scale=1, position=(0, 0, 1.0))
scene += kb.DirectionalLight(
    name="sun", position=(-1, -0.5, 3), look_at=(0, 0, 0), intensity=1.5
)
scene += kb.PerspectiveCamera(name="camera", position=(3, -1, 4), look_at=(0, 0, 1))

rng = np.random.default_rng()
spawn_region = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
for i in range(8):
    velocity = rng.uniform([-1, -1, 0], [1, 1, 0])
    obj = asset_source.create(rng.choice(asset_source.all_asset_ids), velocity=rng.normal(size=3))
    scene += obj
    kb.move_until_no_overlap(obj, simulator, spawn_region=spawn_region)

# --- render (and save the blender file)
renderer.save_state("output/helloworld.blend")
frame = renderer.render_still()

# --- save the output as pngs
kb.write_png(frame["rgba"], "output/helloworld.png")
kb.write_palette_png(frame["segmentation"], "output/helloworld_segmentation.png")
scale = kb.write_scaled_png(frame["depth"], "output/helloworld_depth.png")
logging.info("Depth scale: %s", scale)
