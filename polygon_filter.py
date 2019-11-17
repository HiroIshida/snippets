#!/usr/bin/env python
import rospy
import copy
import numpy as np
from std_msgs.msg import Header
from jsk_recognition_msgs.msg import PolygonArray
from geometry_msgs.msg import Polygon

rospy.init_node("polygon_filter")
pub = rospy.Publisher("topic", PolygonArray, queue_size=1)

def get_polygon_normal(polygon):
    pts = [np.array([pt.x, pt.y, pt.z]) for pt in polygon.points]
    N_pts = len(pts)
    pt0 = pts[0]
    pt1 = pts[N_pts/3]
    pt2 = pts[2 * N_pts/3]

    normalize = lambda x: x/np.linalg.norm(x)
    vec1 = normalize(pt1 - pt0)
    vec2 = normalize(pt2 - pt1)
    vec_normal = np.cross(vec1, vec2)
    return vec_normal

def filter_polygons(polygon_stamped_list, nvec, eps = 0.1):
    isValid = lambda polygon_stamped: \
            np.inner(get_polygon_normal(polygon_stamped.polygon), nvec) < eps
    polygons_filtered = filter(isValid, polygon_stamped_list)
    return polygons_filtered

def callback(msg):
    print(len(msg.polygons))
    nvec = np.array([0, 0, 1])
    msg.polygons = filter_polygons(msg.polygons, nvec)
    print(len(msg.polygons))
    pub.publish(msg)

sub = rospy.Subscriber('/core/multi_plane_estimate/output_polygon', PolygonArray, callback)
rospy.spin()

