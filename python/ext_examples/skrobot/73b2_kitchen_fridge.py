from dataclasses import dataclass
from skrobot.model.primitives import Box
from skrobot.coordinates import CascadedCoords
from typing import List, Optional, Tuple, ClassVar
from skrobot.viewers import TrimeshSceneViewer
import numpy as np


@dataclass
class FridgeParameter:
    W: float = 0.54
    D: float = 0.53
    upper_H: float = 0.65
    container_w: float = 0.48
    container_h: float = 0.62
    container_d: float = 0.45
    panel_d: float = 0.29
    panel_t: float = 0.01
    panel_hights: Tuple[float, ...] = (0.14, 0.285, 0.48)
    door_D = 0.05
    lower_H = 0.81
    joint_x = -0.035
    joint_y = -0.015
    t_bump = 0.02
    d_bump = 0.06


class FridgeModel(CascadedCoords):
    param: FridgeParameter
    links: List[Box]
    regions: List[Box]
    color: ClassVar[Tuple[int, ...]] = (240, 240, 225, 255)

    def __init__(self, param: Optional[FridgeParameter] = None):

        super().__init__()
        if param is None:
            param = FridgeParameter()

        # define upper container
        t_side = 0.5 * (param.W - param.container_w)
        upper_container_co = CascadedCoords()
        side_panel_left = Box([param.container_d, t_side, param.upper_H], pos=(0.5 * param.container_d, 0.5 * param.W - 0.5 * t_side, 0.5 * param.upper_H), face_colors=self.color)
        side_panel_right = Box([param.container_d, t_side, param.upper_H], pos=(0.5 * param.container_d, -0.5 * param.W + 0.5 * t_side, 0.5 * param.upper_H), face_colors=self.color)
        upper_container_co.assoc(side_panel_left, relative_coords="world")
        upper_container_co.assoc(side_panel_right, relative_coords="world")

        t_top = (param.upper_H - param.container_h)
        top_panel = Box([param.container_d, param.container_w, t_top], pos=(0.5 * param.container_d, 0.0, param.upper_H - 0.5 * t_top), face_colors=self.color)
        upper_container_co.assoc(top_panel, relative_coords="world")

        t_back = (param.D - param.container_d)
        back_panel = Box([t_back, param.W, param.upper_H], pos=(param.D - 0.5 * t_back, 0.0, 0.5 * param.upper_H), face_colors=self.color)
        upper_container_co.assoc(back_panel, relative_coords="world")

        links = [side_panel_left, side_panel_right, top_panel, back_panel]

        for panel_h in param.panel_hights:
            panel = Box([param.panel_d, param.container_w, param.panel_t], pos=(param.container_d - 0.5 * param.panel_d, 0.0, panel_h), face_colors=self.color)
            upper_container_co.assoc(panel, relative_coords="world")
            links.append(panel)

        # define regions
        regions = []
        tmp = np.array([0.0] + list(param.panel_hights) + [param.container_h])
        lowers, uppers = tmp[:-1], tmp[1:]
        region_color = (255, 0, 0, 100)
        for lower, upper in zip(lowers, uppers):
            region = Box([param.panel_d, param.container_w, upper - lower], pos=(param.container_d - 0.5 * param.panel_d, 0.0, lower + 0.5 * (upper - lower)), face_colors=region_color)
            upper_container_co.assoc(region, relative_coords="world")
            regions.append(region)

        # define joint
        joint = CascadedCoords(pos=(param.joint_x, -0.5 * param.W + param.joint_y, 0.0))
        upper_container_co.assoc(joint, relative_coords="world")

        # define door
        door = Box([param.door_D, param.W, param.upper_H], pos=(-0.5 * param.door_D, 0.0, 0.5 * param.upper_H), face_colors=self.color)
        bump_left = Box([param.d_bump, param.t_bump, param.container_h], pos=(+0.5 * param.d_bump, 0.5 * param.container_w - 0.5 * param.t_bump, 0.5 * param.container_h), face_colors=self.color)
        bump_right = Box([param.d_bump, param.t_bump, param.container_h], pos=(+0.5 * param.d_bump, -0.5 * param.container_w + 0.5 * param.t_bump, 0.5 * param.container_h), face_colors=self.color)
        joint.assoc(door, relative_coords="world")
        door.assoc(bump_left, relative_coords="world")
        door.assoc(bump_right, relative_coords="world")
        joint.rotate(1.3, "z")

        links.append(door)
        links.append(bump_left)
        links.append(bump_right)

        # define lower box
        lower_box = Box([param.D, param.W, param.lower_H], pos=(0.5 * param.D, 0.0, -0.5 * param.lower_H), face_colors=self.color)
        lower_box.assoc(upper_container_co, relative_coords="world")
        links.append(lower_box)
        lower_box.translate([0, 0, param.lower_H])

        # define base
        self.assoc(lower_box, relative_coords="world")

        self.param = param
        self.links = links
        self.regions = regions


model = FridgeModel()

v = TrimeshSceneViewer()
for link in model.links:
    v.add(link)
for region in model.regions:
    v.add(region)
v.show()
import time; time.sleep(1000)
