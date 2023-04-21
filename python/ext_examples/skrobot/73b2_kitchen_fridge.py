from skrobot.viewers import TrimeshSceneViewer
from skrobot.models.urdf import RobotModelFromURDF
from skrobot.sdf.signed_distance_function import UnionSDF
from skrobot.model.primitives import Box

model = RobotModelFromURDF(urdf_file="/home/h-ishida/tmp/room73b2-hitachi-fiesta-refrigerator/model.urdf")

d = 0.6
w = 0.55
t = 0.03
h_main = 0.6
h_fridge_base = 0.85
fridge_base = Box([0.6, 0.55, h_fridge_base])
fridge_base.translate([+0.03, 0, 0.5 * h_fridge_base])
# fridge_base.visual_mesh.visual.face_colors = [255, 0, 0, 120]

wall1 = Box([d, t, h_main])
wall1.translate([+0.03, 0.5 * w - 0.02, h_fridge_base + 0.5 * h_main])

wall2 = Box([d, t, h_main])
wall2.translate([+0.03, -0.5 * w + 0.02, h_fridge_base + 0.5 * h_main])

wall_top = Box([d, w, 1.5 * t])
wall_top.translate([+0.03, 0, h_fridge_base + h_main])

wall_back = Box([t, w, h_main])
wall_back.translate([0.05 - 0.5 * d, 0.0, h_fridge_base + 0.5 * h_main])

d_plate = 0.3
plate1 = Box([d_plate, w, 0.02])
plate1.translate([-0.1, 0, h_fridge_base + 0.08])

plate2 = Box([d_plate, w, 0.02])
plate2.translate([-0.1, 0, h_fridge_base + 0.24])

plate3 = Box([d_plate, w, 0.02])
plate3.translate([-0.1, 0, h_fridge_base + 0.42])

door = Box([0.2, w, h_main])
door.translate([2.0, 0.0, 0.0])


model.DOOR1.joint_angle(1.5)
vis = TrimeshSceneViewer()
# vis.add(model)
vis.add(fridge_base)
vis.add(wall1)
vis.add(wall2)
vis.add(wall_top)
vis.add(wall_back)
vis.add(plate1)
vis.add(plate2)
vis.add(plate3)
vis.show()
import time
time.sleep(100)
