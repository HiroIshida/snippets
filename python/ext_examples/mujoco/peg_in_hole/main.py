from skrobot.model.primitives import Box
from skrobot.viewers import TrimeshSceneViewer
from mujoco_xml_editor import MujocoXmlEditor
import mujoco
import mujoco_viewer

base_plate = Box([0.2, 0.2, 0.01])
base_plate.translate([0.0, 0.0, 0.005])

hole_width = 0.03

left_plate = Box([0.2, (0.2 - hole_width) * 0.5, 0.1])
left_plate.translate([0.0, 0.1 - (0.2 - hole_width) * 0.25, 0.05])
right_plate = Box([0.2, (0.2 - hole_width) * 0.5, 0.1])
right_plate.translate([0.0, -0.1 + (0.2 - hole_width) * 0.25, 0.05])

hole_depth = 0.06
front_plate = Box([(0.2 - hole_depth) * 0.5, 0.2, 0.1])
front_plate.translate([0.1 - (0.2 - hole_depth) * 0.25, 0.0, 0.05])
back_plate = Box([(0.2 - hole_depth) * 0.5, 0.2, 0.1])
back_plate.translate([-0.1 + (0.2 - hole_depth) * 0.25, 0.0, 0.05])

primitives = [base_plate, left_plate, right_plate, front_plate, back_plate]
for p in primitives:
    p.translate([0.0, 0.0, 0.0])


# stick
eps = 0.005
stick = Box([hole_depth - eps, hole_width - eps, 0.15])
stick.translate([0.0, 0.0, 0.3])

editor = MujocoXmlEditor.empty("peg_in_hole.xml")
editor.add_primitive_composite(primitives[0], primitives, "hole_plate")
editor.add_primitive(stick, "stick")
editor.add_sky()
editor.add_ground()
editor.add_light()
xmlstr = editor.to_string()

model = mujoco.MjModel.from_xml_string(xmlstr)
data = mujoco.MjData(model)
viewer = mujoco_viewer.MujocoViewer(model, data)

while True:
    viewer.render()
    mujoco.mj_step(model, data)

# show 
# viewer = TrimeshSceneViewer()
# viewer.add(base_plate)
# viewer.add(left_plate)
# viewer.add(right_plate)
# viewer.add(front_plate)
# viewer.add(back_plate)
# viewer.show()
# import time; time.sleep(1000)
