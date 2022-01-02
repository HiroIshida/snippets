import mujoco_py
from mujoco_py import MjSim, MjViewer
from mujoco_py.modder import TextureModder
import os
import time

mj_path = mujoco_py.utils.discover_mujoco()
xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
model = mujoco_py.load_model_from_path(xml_path)
sim = MjSim(model)
viewer = MjViewer(sim)
modder = TextureModder(sim)

viewer.render()
#time.sleep(5)

t = 0
while True:
    sim.step()
    viewer.render()
    t += 1
    if t > 1000: break
