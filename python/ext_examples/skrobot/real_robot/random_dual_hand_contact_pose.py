import argparse
import skrobot
import numpy as np
from skrobot.models.pr2 import PR2
from skrobot.interfaces.ros import PR2ROSRobotInterface
from skrobot.coordinates import Coordinates
from skrobot.viewers import TrimeshSceneViewer

# argparse to determine random seed
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, help="seed", default=0)
args = parser.parse_args()
np.random.seed(args.seed)

pr2 = PR2()
while True:
    pr2.reset_manip_pose()
    x = np.random.uniform(0.3, 0.7)
    y = np.random.uniform(-0.3, 0.3)
    z = np.random.uniform(0.5, 0.9)
    y_abs_from_center = 0.02

    target_coords = skrobot.coordinates.Coordinates([x, y-y_abs_from_center, z], [np.pi * 0.5, 0, 0.0])
    res = pr2.inverse_kinematics(
        target_coords,
        link_list=pr2.rarm.link_list,
        move_target=pr2.rarm_end_coords)
    if res is False:
        continue


    target_coords = skrobot.coordinates.Coordinates([x, y+y_abs_from_center, z], [-np.pi * 0.5, 0, 0.0])
    res = pr2.inverse_kinematics(
        target_coords,
        link_list=pr2.larm.link_list,
        move_target=pr2.larm_end_coords)
    if res is False:
        continue
    break

print(res)
v = TrimeshSceneViewer()
v.add(pr2)
v.show()
ri = PR1ROSRobotInterface(pr2)
ri.angle_vector(pr2.angle_vector())
import time; time.sleep(1000)
