import tqdm
import numpy as np
from skrobot.models.urdf import RobotModelFromURDF
from skrobot.model.robot_model import RobotModel
from skrobot.coordinates import Coordinates
from skrobot.viewers import TrimeshSceneViewer
from skrobot.model.primitives import Axis
import logging
from typing import Callable, Optional

def solve_ik(robot: RobotModel, co: Coordinates) -> bool:
    ret = robot.inverse_kinematics(
        co,
        move_target=robot.r_gripper_tool_frame,
        rotation_axis=True,
        stop=100,
    )
    return ret is not False  # ret may be angle

robot: RobotModel = RobotModelFromURDF(urdf_file="./gripper.urdf")

for _ in tqdm.tqdm(range(30)):
    pos = np.random.uniform(-1, 1, size=3)
    rot = np.random.uniform(-1, 1, size=3)
    co = Coordinates(pos, rot)
    ret = solve_ik(robot, co)
    assert ret


v = TrimeshSceneViewer()
v.add(robot)
ax = Axis.from_coords(co)
v.add(ax)
v.show()
import time; time.sleep(1000)
