from pathlib import Path
from robot_descriptions.robotiq_2f85_mj_description import PACKAGE_PATH
from mujoco_xml_editor import MujocoXmlEditor
import mujoco

hand_xml_path = Path(PACKAGE_PATH) / "2f85.xml"
editor = MujocoXmlEditor.load(hand_xml_path)
xmlstr = editor.to_string()
model = mujoco.MjModel.from_xml_string(xmlstr)

# The goal is finding a right follower geom index
# The right follower geom is included in "right_follower" body

body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_follower")
assert body_id != -1
assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) == "right_follower"

start_geom_addr = model.body_geomadr[body_id]
assert start_geom_addr != -1
geom_count = model.body_geomnum[body_id]

collision_geom_idx = None
for i in range(geom_count):
    geom_id = start_geom_addr + i
    group = model.geom_group[geom_id]
    # group == 2 for visual, 3 for collision
    if group == 3:
        collision_geom_id = geom_id
        break
print(f"Collision geom id: {collision_geom_id}")

