from skrobot.model.primitives import Box, Axis
from skrobot.coordinates import Coordinates
from skrobot.coordinates.base import transform_coords
from skrobot.viewers import PyrenderViewer

table = Box([0.6, 1.0, 0.05])
table.translate([0.0, 0.0, 0.7])

table_surface_center = Axis.from_coords(table.copy_worldcoords())
table_surface_center.translate([0.0, 0.0, 0.025])

box = Box([0.1, 0.1, 0.1], face_colors=[255, 0, 0, 255])
box.newcoords(table_surface_center.copy_worldcoords())
box.translate([0.0, 0.0, 0.05])
box.translate([0.3, 0.5, 0.0])

tf_table_2_world = table_surface_center.get_transform()
tf_box_2_world = box.get_transform()
tf_box_2_table = tf_box_2_world * tf_table_2_world.inverse_transformation()
print(tf_box_2_table.translation)

v = PyrenderViewer()
v.add(table)
v.add(table_surface_center)
v.add(box)
v.show()
import time; time.sleep(1000)
