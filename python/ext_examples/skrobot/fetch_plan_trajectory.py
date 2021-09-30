import skrobot
import pybullet
import pybullet as pb
import numpy as np
import time
from skrobot.interfaces import PybulletRobotInterface
from skrobot.utils import sdf_box

def create_box(center, b, margin=0.0):
    quat = [0, 0, 0, 1]
    sdf = lambda X: sdf_box(X, b, center) - margin
    return sdf

try:
    fetch
except:
    fetch = skrobot.models.Fetch()
    viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
    viewer.add(fetch)
    viewer.show()

    box_width = np.array([0.5, 0.3, 0.6])
    box_center = np.array([0.9, -0.2, 0.9])
    sdf = create_box(box_center, box_width*0.5, margin=0.1)
    box = skrobot.models.Box(
        extents=box_width, face_colors=(1., 0, 0)
    )
    box.translate(box_center)
    viewer.add(box)

link_list = [fetch.link_list[3]] + fetch.link_list[6:13]
joint_list = [link.joint for link in link_list]
set_joint_angles = lambda av: [j.joint_angle(a) for j, a in zip(joint_list, av)]
get_joint_angles = lambda : np.array([j.joint_angle() for j in joint_list])

av_init = [0.0, 0.58, 0.35, -0.74, -0.70, -0.17, -0.63, 0.0]
set_joint_angles(av_init)

target_coords = skrobot.coordinates.Coordinates([0.6, -0.7, 1.0], [0, 0, 0])
collision_coords_list = [skrobot.coordinates.CascadedCoords(
    parent=link) for link in [fetch.r_gripper_finger_link, fetch.wrist_roll_link, fetch.elbow_flex_link]]

traj = fetch.plan_trajectory(target_coords, 10, link_list, fetch.end_coords,
        collision_coords_list, sdf,
        rot_also=True
        )

# check if generated path is collision free
def get_endpos(av):
    set_joint_angles(av)
    x = fetch.end_coords[0].worldpos()
    return x
X = np.array([get_endpos(av) for av in traj])
assert np.all(sdf(X) > 0)



print("solved")

import time 
time.sleep(3)
counter = 0
for av in traj:
    time.sleep(1.0)
    set_joint_angles(av)
    viewer.redraw()
    viewer.save_image("img{0}.png".format(counter))
    counter += 1

