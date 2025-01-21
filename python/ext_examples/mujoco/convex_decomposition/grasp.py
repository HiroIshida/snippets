import xml.etree.ElementTree as ET
from pathlib import Path
import coacd
import trimesh
import ycb_utils
import time
import numpy as np
import math
import mujoco
import mujoco_viewer
import ycb_utils
from pathlib import Path
from robot_descriptions.robotiq_2f85_mj_description import PACKAGE_PATH
import xml.etree.ElementTree as ET
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import rpy2quaternion



def get_modified_xml() -> str:
    hand_xml_path = Path(PACKAGE_PATH) / "2f85.xml"
    root = ET.parse(hand_xml_path).getroot()
    asset = root.find("asset")
    worldbody = root.find("worldbody")

    # replace the file paths in the xml with absolute paths
    for mesh in asset.findall("mesh"):
        file_path_str = mesh.get("file")
        abs_file_path = Path(PACKAGE_PATH) / "assets" / file_path_str
        mesh.set("file", str(abs_file_path))

    # add headlight
    if root.find("visual") is None:
        visual = ET.Element("visual")
        visual.append(ET.Element("headlight", {"diffuse": "0.6 0.6 0.6", "ambient": "0.3 0.3 0.3", "specular": "0 0 0"}))
        visual.append(ET.Element("rgba", {"haze": "0.15 0.25 0.35 1"}))
        visual.append(ET.Element("global", {"azimuth": "150", "elevation": "-20"}))
        root.append(visual)

    # add skybox and groundplane
    asset.append(ET.Element("texture", {"type": "skybox", "builtin": "gradient", "rgb1": "0.3 0.5 0.7", "rgb2": "0 0 0", "width": "512", "height": "3072"}))
    asset.append(ET.Element("texture", {"type": "2d", "name": "groundplane", "builtin": "checker", "mark": "edge", "rgb1": "0.2 0.3 0.4", "rgb2": "0.1 0.2 0.3", "markrgb": "0.8 0.8 0.8", "width": "300", "height": "300"}))
    asset.append(ET.Element("material", {"name": "groundplane", "texture": "groundplane", "texuniform": "true", "texrepeat": "5 5", "reflectance": "0.2"}))
    worldbody.append(ET.Element("geom", {"name": "floor", "size": "0 0 0.05", "type": "plane", "material": "groundplane"}))

    # mocap body
    mocap_body = ET.Element("body", {"name": "arm", "mocap": "true", "pos": "0 0 0.0"})
    mocap_body.append(ET.Element("geom", {"type": "box", "size": "0.02 0.03 0.03", "contype": "0", "conaffinity": "0"}))
    worldbody.append(mocap_body)

    assert root.find("equality") is not None
    equality = root.find("equality")
    weld = ET.Element("weld", {"body1": "arm", "body2": "base_mount"})
    equality.append(weld)

    # add free joint to base_mount 
    for body in worldbody.findall("body"):
        if body.get("name") == "base_mount":
            body.append(ET.Element("joint", {"type": "free"}))

    # add mesh
    dir_path = Path("parts")
    if not dir_path.exists():
        dir_path.mkdir()
        m = ycb_utils.load("019_pitcher_base")
        mesh = coacd.Mesh(m.vertices, m.faces)
        parts = coacd.run_coacd(mesh, threshold=0.02)
        for i, (V, F) in enumerate(parts):
            file_path = dir_path / f"part_{i}.obj"
            trimesh.Trimesh(V, F).export(file_path)

    asset = root.find("asset")
    for part_file in sorted(dir_path.glob("part_*.obj")):
        mesh_name = part_file.stem
        ET.SubElement(asset, 'mesh', attrib={
            'file': part_file.resolve().expanduser().as_posix(),
            'name': mesh_name
        })

    pitcher_body = ET.SubElement(worldbody, 'body', attrib={
        'name': 'pitcher_body',
        'pos': '0.3 0 0',
        'euler': '0 0 2.3707'
    })
    ET.SubElement(pitcher_body, 'joint', attrib={
        'name': 'pitcher_joint',
        'type': 'free'
    })

    for part_file in sorted(dir_path.glob("part_*.obj")):
        mesh_name = part_file.stem
        ET.SubElement(pitcher_body, 'geom', attrib={
            'mesh': mesh_name,
            'name': f"{mesh_name}_geom",
            'type': 'mesh',
            'density': '10',
        })
    return ET.tostring(root, encoding="unicode", method="xml")

def set_mocap_position(data: mujoco.MjData, pos: np.ndarray, hand_also: bool = False):
    pos = np.array(pos)
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.array([0.0, 0.5 * np.pi, 0.5 * np.pi]), "xyz")
    if hand_also:
        data.qpos[0:3] = pos
        data.qpos[3:7] = quat
    data.mocap_pos[0] = pos
    data.mocap_quat[0] = np.array(quat)

if __name__ == "__main__":
    xml_str = get_modified_xml()
    xml_path = "pitcher_decomposed.xml"
    with open(xml_path, "w") as f:
        f.write(xml_str)
    print(f"MuJoCo XML file generated: {xml_path}")

    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    finger_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")

    input("press enter to start the simulation")
    pos = np.array([0.0, -0.02, 0.15])
    # very important that the hand is also moved to the same position
    set_mocap_position(data, pos, hand_also=True)

    # move the finger down
    for i in range(110):
        pos[0] += 0.0004
        set_mocap_position(data, pos)
        data.ctrl[finger_act_id] = 100
        mujoco.mj_step(model, data)
        print(data.qpos[:7])
        viewer.render()
    print("done1")

    # grasp
    for i in range(300):
        data.ctrl[finger_act_id] = 220
        mujoco.mj_step(model, data)
        viewer.render()
    print("done2")

    # move the finger up
    for i in range(300):
        pos[2] += 0.0004
        set_mocap_position(data, pos)
        mujoco.mj_step(model, data)
        viewer.render()

    while True:
        mujoco.mj_step(model, data)
        viewer.render()
