# NOTE please checkout to urdf_dev (e2e4e2b)
import copy

import xml.etree.ElementTree as ET
from xml.dom import minidom
import skrobot
from skrobot.model import Box
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

def create_xml_link(name, geom_list):
    if not isinstance(geom_list, list):
        geom_list = [geom_list]

    xml_link = ET.Element('link', {'name': name})
    for geom in geom_list:
        xml_link.append(create_xml_visual(geom))
        xml_link.append(create_xml_collision(geom))
    return xml_link

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

b_lower = Box([width, width, thickness])
b_upper = copy.copy(b_lower)
b_upper.translate([0, 0, height])

b_side = Box([width, thickness, height])
b_side_left = copy.copy(b_side)
b_side_left.translate([0, 0.5 * width, 0.5 * height])
b_side_right = copy.copy(b_side)
b_side_right.translate([0, -0.5 * width, 0.5 * height])

b_back = Box([thickness, width, height])
b_back.translate([0.5 * width, 0, 0.5 * height])


geom_list = [b_lower, b_upper, b_side_left, b_side_right, b_back]
viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
for geom in geom_list:
    viewer.add(geom)

viewer.show()


xml_link = create_xml_link("base_link", geom_list)


xml_robot = create_robot("myrobot", [xml_link])
xmlstr = minidom.parseString(ET.tostring(xml_robot)).toprettyxml(indent="   ")

with open('./test.xml', 'w') as f:
    f.write(xmlstr)

"""
import skrobot
robot = skrobot.models.urdf.RobotModelFromURDF(urdf_file="./test.xml")
link = robot.base_link
viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
viewer.add(robot)
viewer.show()
"""
