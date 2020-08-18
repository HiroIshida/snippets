#!/usr/bin/env python
import rospy
from nav_msgs.msg import OccupancyGrid
import tf
import copy
import StringIO
import numpy as np
import matplotlib.pyplot as plt 
import pickle 

def gridmsg2nparray(msg):
    info = msg.info
    w = info.width
    h = info.height
    npnized = np.array(msg.data).reshape((w, h))
    return npnized

class MapData:
    def __init__(self, arr, res, origin, tf_base_to_odom):
        self.arr = arr
        self.tf_base_to_odom = tf_base_to_odom
        self.origin = origin
        self.res = res

class MapManager:
    def __init__(self):
        msg_name  = "/move_base_node/local_costmap/costmap"
        self.sub = rospy.Subscriber(msg_name, OccupancyGrid, self.map_callback)
        self.listener = tf.TransformListener()
        self.msg = None
        self.arr = None
        self.data = None

    def map_callback(self, msg):
        print("rec")
        self.msg = copy.deepcopy(msg)
        info = msg.info
        arr = gridmsg2nparray(msg)
        self.arr = np.fliplr(arr)

        while(True):
            try:
                tf_base_to_odom = self.listener.lookupTransform(msg.header.frame_id, "/base_footprint", rospy.Time(0))
                print(tf_base_to_odom)
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
        print(info.origin)
        self.data = MapData(np.fliplr(arr), info.resolution, info.origin, tf_base_to_odom)

    def show_map(self):
        plt.pcolor(self.arr)

    def save_map(self, name="localcost_pr2.pickle"):
        with open(name, 'wb') as f:
            pickle.dump(self.data, f)


if __name__=='__main__':
    rospy.init_node('map_saver')
    mm = MapManager();
    r = rospy.Rate(10)
    for i in range(20):
        r.sleep()
    mm.show_map()
    plt.show()
    mm.save_map()

