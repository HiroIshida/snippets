#!/usr/bin/env python
import scipy.interpolate  
import rospy
from nav_msgs.msg import OccupancyGrid
import tf
import copy
import numpy as np
import matplotlib.pyplot as plt 
import dill 

def quaternion_matrix(quaternion): # fetched from tf.transformations
    _EPS = np.finfo(float).eps * 4.0
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)

def generate_sdf(msg, tf_base_to_odom):
    info = msg.info
    n_grid = np.array([info.width, info.height])
    res = info.resolution
    origin = info.origin

    b_min = np.zeros(2)
    b_max = n_grid * res
    xlin, ylin = [np.linspace(b_min[i], b_max[i], n_grid[i]) for i in range(2)]

    tmp = np.array(msg.data).reshape((n_grid[1], n_grid[0])) # about to be transposed!!
    arr = tmp.T # [IMPORTANT] !!
    fp_wrt_map = scipy.interpolate.RegularGridInterpolator((xlin, ylin), arr, 
            method='linear', bounds_error=True, fill_value=0.0) 

    def base_to_map(P): 
        n_points = len(P)
        pos_base_to_odom, rot_base_to_odom = tf_base_to_odom
        #M = tf.transformations.quaternion_matrix(rot_base_to_odom)[:2, :2]
        M = quaternion_matrix(rot_base_to_odom)[:2, :2]
        points = P.dot(M.T) + np.repeat(
                np.array([[pos_base_to_odom[0], pos_base_to_odom[1]]]), n_points, 0)

        mappos_origin = np.array([[origin.position.x, origin.position.y]])
        points = points - np.repeat(mappos_origin, n_points, 0)
        return points

    fp_wrt_base = lambda X: fp_wrt_map(base_to_map(X))
    return fp_wrt_base

class MapManager:
    def __init__(self):
        msg_name  = "/move_base_node/local_costmap/costmap"
        self.sub = rospy.Subscriber(msg_name, OccupancyGrid, self.map_callback)
        self.listener = tf.TransformListener()
        self.msg = None
        self.arr = None
        self.costmapf = None

    def map_callback(self, msg):
        print("rec")
        self.msg = copy.deepcopy(msg)
        while(True):
            try:
                tf_base_to_odom = self.listener.lookupTransform(msg.header.frame_id, "/base_footprint", rospy.Time(0))
                print(tf_base_to_odom)
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
        self.costmapf = generate_sdf(msg, tf_base_to_odom)

    def show_map_wrtbase(self):
        b = 1.0
        xlin = np.linspace(-b, b, 200)
        ylin = np.linspace(-b, b, 200)
        X, Y = np.meshgrid(xlin, ylin)
        pts = np.array(list(zip(X.flatten(), Y.flatten())))
        Z_ = self.costmapf(pts)
        Z = Z_.reshape((200, 200))
        fig, ax = plt.subplots()
        ax.contourf(X, Y, Z)
        plt.show()

    def save_pickle(self):
        with open("costmapf.dill", "wb") as f:
            dill.dump(self.costmapf, f)

if __name__=='__main__':
    rospy.init_node('map_saver')
    mm = MapManager();
    r = rospy.Rate(10)
    for i in range(10):
        r.sleep()
    mm.show_map_wrtbase()
    mm.save_pickle()
