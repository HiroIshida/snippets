#!/usr/bin/env python
import time
import numpy as np
import scipy.interpolate as itp

import rospy
import tf 
from nav_msgs.msg import OccupancyGrid

def quaternion_matrix(quaternion): # fetched from tf.transformations (ver 2009)
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

class CostMapManager(object):
    def __init__(self, 
            tf_listener,
            world_frame='map', 
            costmap_topic_name='/move_base_node/local_costmap/costmap'):

        self._tf_listener = tf_listener
        self._world_frame = world_frame
        rospy.Subscriber(costmap_topic_name, OccupancyGrid, self._costmap_callback)

        # these two are obtained at the same moment
        self._map_msg = None
        self._tf_world_to_costmap = None

    def _costmap_callback(self, msg):
        print("callback called")
        counter = 0
        while True:
            try:
                self._tf_world_to_costmap = self._tf_listener.lookupTransform(msg.header.frame_id, self._world_frame, rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                time.sleep(0.2)
                counter += 1
            if counter == 5:
                print("abort tf lookup")
                return 

        self._map_msg = msg

    def _convert_world_to_map(self, points):
        n_points = len(points)
        map_origin = self._map_msg.info.origin
        mappos_origin = np.array([[map_origin.position.x, map_origin.position.y]])

        trans, rot = self._tf_world_to_costmap
        M = quaternion_matrix(rot)[:2, :2]
        points_converted_tmp = points.dot(M.T) + np.repeat(
                np.array([[trans[0], trans[1]]]), n_points, 0)
        points_converted = points_converted_tmp - np.repeat(mappos_origin, n_points, 0)
        return points_converted

    def _create_map_interpolator(self):
        info = self._map_msg.info
        N_grid = np.array([info.width, info.height])
        b_min = np.zeros(2)
        b_max = N_grid * info.resolution
        xlin, ylin = [np.linspace(b_min[i], b_max[i], N_grid[i]) for i in range(2)]

        data_arr = np.array(self._map_msg.data).reshape(N_grid).T
        f_interp = itp.RegularGridInterpolator((xlin, ylin), data_arr, bounds_error=False, fill_value=np.inf)
        return f_interp, xlin, ylin

    def compute_cost(self, points):
        # these points must be w.r.t. the world frame
        points_converted = self._convert_world_to_map(points)
        itp, xlin, ylin = self._create_map_interpolator()
        costs = itp(points_converted)
        return costs

    def debug_plot(self):
        N_grid = np.array([100, 100])
        b_min = -np.ones(2) * 3
        b_max = +np.ones(2) * 3
        xlin, ylin = [np.linspace(b_min[i], b_max[i], N_grid[i]) for i in range(2)]
        grids = np.meshgrid(xlin, ylin)
        pts = np.array(zip(*[g.flatten() for g in grids]))
        vals = self.compute_cost(pts).reshape(N_grid)

        import matplotlib.pyplot as plt
        plt.contourf(grids[0], grids[1], vals)
        plt.colorbar()
        plt.show()

if __name__=='__main__':

    rospy.init_node('overlap_finder')
    tf_listener = tf.TransformListener()
    cmm = CostMapManager(tf_listener, world_frame="base_link")
    rospy.spin()
    #cmm.debug_plot()
