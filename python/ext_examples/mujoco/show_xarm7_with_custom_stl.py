import os
import numpy as np
import mujoco
import mujoco_viewer
# from robot_descriptions.loaders.mujoco import load_robot_description
from robot_descriptions.xarm7_mj_description import PACKAGE_PATH
from pathlib import Path
import xml.etree.ElementTree as ET
import ycb_utils

scene_path = Path(PACKAGE_PATH) / "scene.xml"

# modify the scene.xml to include the mesh of the object
tree = ET.parse(scene_path)
asset = tree.find("asset")
assert asset is not None
mesh_path = str(ycb_utils.resolve_path("019_pitcher_base"))
asset_name = "pitcher_base"
mesh = ET.SubElement(asset, "mesh", attrib={"file": mesh_path, "name": asset_name})

worldbody = tree.find("worldbody")
assert worldbody is not None
obj = ET.SubElement(worldbody, "geom")
obj.set("type", "mesh")
obj.set("mesh", asset_name)
obj.set("pos", "0.5 0 0.0")
obj.set("euler", "0 0 0.8")

xml_str = ET.tostring(tree.getroot(), encoding="unicode")

print(xml_str)

# to resolve the relative path of robot description
original_cwd = os.getcwd()
try:
    os.chdir(scene_path.parent)
    model = mujoco.MjModel.from_xml_string(xml_str)
finally:
    os.chdir(original_cwd)
data = mujoco.MjData(model)

viewer = mujoco_viewer.MujocoViewer(model, data)

desired_qpos1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 255])
desired_qpos2 = np.array([0.0, -0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0])
desired_qpos3 = np.array([0.0, -1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 3])
desired_list = [desired_qpos1, desired_qpos2, desired_qpos3]
current_index = 0
desired = desired_list[current_index]

kp = 100
kd = 5

input("press enter to start the simulation")

for _ in range(10000):
    print(current_index)
    if viewer.is_alive:
        current_qpos = data.qpos[:8]
        current_qvel = data.qvel[:8]
        pos_error = desired - current_qpos
        vel_error = -current_qvel
        if np.linalg.norm(pos_error[:7]) < 0.1:
            if current_index < len(desired_list) - 1:
                current_index += 1
                desired = desired_list[current_index]
        control_input = kp * pos_error + kd * vel_error
        data.ctrl[:8] = control_input
        mujoco.mj_step(model, data)
        viewer.render()
    else:
        break

viewer.close()
