#!/usr/bin/env python
import scipy.interpolate  
import rospy
from nav_msgs.msg import OccupancyGrid
import tf
import copy
import numpy as np
import matplotlib.pyplot as plt 
from costmap import CostmapFunctionData

class MapManager:
    def __init__(self):
        msg_name  = "/move_base_node/local_costmap/costmap"
        self.sub = rospy.Subscriber(msg_name, OccupancyGrid, self.map_callback)
        self.listener = tf.TransformListener()
        self.costmapdata = None

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
        self.costmapdata = CostmapFunctionData(msg, tf_base_to_odom)

    def show_map_wrtbase(self):
        costmapf = self.costmapdata.convert2sdf()
        b = 1.0
        xlin = np.linspace(-b, b, 200)
        ylin = np.linspace(-b, b, 200)
        X, Y = np.meshgrid(xlin, ylin)
        pts = np.array(list(zip(X.flatten(), Y.flatten())))
        Z_ = costmapf(pts)
        Z = Z_.reshape((200, 200))
        fig, ax = plt.subplots()
        ax.contourf(X, Y, Z)
        plt.show()

    def save(self):
        self.costmapdata.save()

if __name__=='__main__':
    rospy.init_node('map_saver')
    mm = MapManager();
    r = rospy.Rate(10)
    for i in range(10):
        r.sleep()
    mm.show_map_wrtbase()
    mm.save()
