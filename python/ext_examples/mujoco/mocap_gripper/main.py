import math
import mujoco
import mujoco_viewer
from pathlib import Path
from robot_descriptions.xarm7_mj_description import PACKAGE_PATH
import xml.etree.ElementTree as ET

def get_modified_xml() -> str:
    hand_xml_path = Path(PACKAGE_PATH) / "hand.xml"
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
        visual.append(ET.Element("headlight", {
            "diffuse": "0.6 0.6 0.6",
            "ambient": "0.3 0.3 0.3",
            "specular": "0 0 0"
        }))
        visual.append(ET.Element("rgba", {
            "haze": "0.15 0.25 0.35 1"
        }))
        visual.append(ET.Element("global", {
            "azimuth": "150",
            "elevation": "-20"
        }))
        root.append(visual)

    # add skybox
    asset.append(ET.Element("texture", {
        "type": "skybox",
        "builtin": "gradient",
        "rgb1": "0.3 0.5 0.7",
        "rgb2": "0 0 0",
        "width": "512",
        "height": "3072"
    }))

    # add ground plane
    asset.append(ET.Element("texture", {
        "type": "2d",
        "name": "groundplane",
        "builtin": "checker",
        "mark": "edge",
        "rgb1": "0.2 0.3 0.4",
        "rgb2": "0.1 0.2 0.3",
        "markrgb": "0.8 0.8 0.8",
        "width": "300",
        "height": "300"
    }))

    asset.append(ET.Element("material", {
        "name": "groundplane",
        "texture": "groundplane",
        "texuniform": "true",
        "texrepeat": "5 5",
        "reflectance": "0.2"
    }))

    worldbody.append(ET.Element("geom", {
        "name": "floor",
        "size": "0 0 0.05",
        "type": "plane",
        "material": "groundplane"
    }))


    # make base link mocap
    for body in worldbody.findall("body"):
        if body.get("name") == "xarm_gripper_base_link":
            body.set("mocap", "true")


    return ET.tostring(root, encoding="unicode", method="xml")

if __name__ == "__main__":
    xml_str = get_modified_xml()
    print(xml_str)
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    finger_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")

    input("press enter to start the simulation")
    for i in range(10000):
        if viewer.is_alive:
            mujoco.mj_step(model, data)
            viewer.render()
        else:
            break
        data.mocap_pos[0][2] = 0.05 + 0.05 * math.sin(0.01 * i)
        data.ctrl[finger_act_id] = 127.5 + 127.5 * math.sin(0.01 * i)
    viewer.close()
