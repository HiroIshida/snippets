# NOTE please checkout to urdf_dev (e2e4e2b)
import copy

import xml.etree.ElementTree as ET
from xml.dom import minidom
import skrobot
from skrobot.model import Box, Axis
from skrobot.model import Joint, RotationalJoint, Link
from skrobot.coordinates import rpy_angle

def list2str(lst):
    string = ""
    for idx in range(len(lst)-1):
        string += str(lst[idx]) + " " 
    string += str(lst[-1])
    return string

def _create_xml_visual_like(box, tag='visual'):
    pos = box.worldpos()
    rot = box.worldrot()
    rpy = rpy_angle(rot)[0]

    origin = ET.Element('origin', {'xyz': list2str(pos), 'rpy': list2str(rpy)})
    geometry = ET.Element('geometry')
    xml_box = ET.SubElement(geometry, 'box', {'size': list2str(box._extents)})

    xml_visual = ET.Element(tag)
    xml_visual.append(geometry)
    xml_visual.append(origin)
    return xml_visual

def create_xml_visual(box):
    return _create_xml_visual_like(box)

def create_xml_collision(box):
    return _create_xml_visual_like(box, tag='collision')

def create_xml_link(name, geom_list, with_collision=True):
    if not isinstance(geom_list, list):
        geom_list = [geom_list]

    xml_link = ET.Element('link', {'name': name})
    for geom in geom_list:
        xml_link.append(create_xml_visual(geom))
        if with_collision:
            xml_link.append(create_xml_collision(geom))
    return xml_link

def create_xml_joint(name, jtype, plink_name, clink_name, axis, xyz, rpy,
        effort=1.0, lower=0.0, upper=2.0, velocity=2.0):
    xml_joint = ET.Element('joint', {'name': name, 'type': jtype})
    ET.SubElement(xml_joint, 'parent', {'link': plink_name})
    ET.SubElement(xml_joint, 'child', {'link': clink_name})
    ET.SubElement(xml_joint, 'axis', {'xyz': list2str(axis)})
    ET.SubElement(xml_joint, 'origin', {'rpy': list2str(rpy), 'xyz': list2str(xyz)})
    if not jtype=="fixed":
        ET.SubElement(xml_joint, 'limit', {'effort': str(effort), 'lower': str(lower), 'upper': str(upper), 'velocity': str(velocity)})
    return xml_joint


def create_robot(robot_name, xml_link_list, xml_joint_list=[]):
    root = ET.Element('robot', {'name': robot_name})
    for xml_link in xml_link_list:
        root.append(xml_link)
    for xml_joint in xml_joint_list:
        root.append(xml_joint)
    return root

thickness = 0.03
width = 0.8
height = 2.0

# create base_link
b_lower = Box([width, width, thickness])
b_upper = copy.deepcopy(b_lower)
b_upper.translate([0, 0, height])
b_side = Box([width, thickness, height])
b_side_left = copy.deepcopy(b_side)
b_side_left.translate([0, 0.5 * width, 0.5 * height])
b_side_right = copy.deepcopy(b_side)
b_side_right.translate([0, -0.5 * width, 0.5 * height])
b_back = Box([thickness, width, height])
b_back.translate([0.5 * width, 0, 0.5 * height])
base_link_geom_list = [b_lower, b_upper, b_side_left, b_side_right, b_back]
xml_base_link = create_xml_link("base_link", base_link_geom_list)

# create door_link
b_door = Box([thickness, width, height])
b_door.translate([0.0, 0.5 * width, 0.0])
xml_door_link = create_xml_link("door_link", [b_door])

# create handle_link
b_handle = Box([0.1, 0.02, 0.2])
xml_handle_link = create_xml_link("handle_link", [b_handle], with_collision=False)

xml_door_joint = create_xml_joint("door_joint", "revolute", "base_link", "door_link", [0, 0, 1], xyz=[-0.5*width, -0.5*width, 0.5*height], rpy=[0, 0, 0])
xml_handle_joint = create_xml_joint("handle_joint", "fixed", "door_link", "handle_link", [0, 0, 1], xyz=[-0.03, 0.9*width, 0.25*height], rpy=[0, 0, 0])
xml_robot = create_robot("myrobot", [xml_base_link, xml_door_link, xml_handle_link], [xml_door_joint, xml_handle_joint])
xmlstr = minidom.parseString(ET.tostring(xml_robot)).toprettyxml(indent="   ")
with open('./test.xml', 'w') as f:
    f.write(xmlstr)

viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
robot_model = skrobot.models.urdf.RobotModelFromURDF(urdf_file="./test.xml")
axis = Axis()
robot_model.handle_link.assoc(axis, relative_coords=axis)
viewer.add(robot_model)
viewer.add(axis)
viewer.show()
