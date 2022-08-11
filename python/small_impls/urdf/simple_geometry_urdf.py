import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
from xml.etree.ElementTree import Element, ElementTree, SubElement

import numpy as np


@dataclass
class BoxConfig:
    name: str
    size: Tuple[float, float, float]
    density: float = 2.7 * 10**3

    @property
    def mass(self):
        v = np.prod(self.size)
        return v * self.density

    @property
    def inertia(self):
        a, b, c = self.size
        ixx = 1 / 3.0 * (b**2 + c**2) * self.mass
        iyy = 1 / 3.0 * (c**2 + a**2) * self.mass
        izz = 1 / 3.0 * (a**2 + b**2) * self.mass
        return ixx, iyy, izz


def create_box_urdf(config: BoxConfig) -> Path:
    root = Element("robot", name=config.name)
    link = SubElement(root, "link", name="base_link")

    geometry = Element("geometry")
    SubElement(geometry, "box", size="{} {} {}".format(*config.size))

    # create visual
    visual = Element("visual")
    visual.append(geometry)
    link.append(visual)

    # create collision
    collision = Element("collision")
    collision.append(geometry)
    link.append(collision)

    # create inertial
    inertial = Element("inertial")
    SubElement(inertial, "origin", rpy="0 0 0", xyz="0 0 0")
    SubElement(inertial, "mass", value=str(config.mass))
    ixx, iyy, izz = config.inertia
    SubElement(
        inertial,
        "inertia",
        ixx=str(ixx),
        ixy="0.0",
        ixz="0.0",
        iyy=str(iyy),
        iyz="0.0",
        izz=str(izz),
    )
    link.append(inertial)

    tree = ElementTree(root)
    directory_path = Path("/tmp/auto_generated_urdf")
    directory_path.mkdir(exist_ok=True)
    file_path = directory_path / "{}.urdf".format(uuid.uuid4())
    tree.write(str(file_path))
    return file_path


p = create_box_urdf(BoxConfig("hoge", (0.1, 0.2, 0.3)))
