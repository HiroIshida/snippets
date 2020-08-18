#!/usr/bin/env python
import rospy
from nav_msgs.msg import OccupancyGrid
import tf
import copy
import StringIO
import numpy as np
import matplotlib.pyplot as plt 

def gridmsg2nparray(msg):
    info = msg.info
    w = info.width
    h = info.height
    npnized = np.array(msg.data).reshape((w, h))
    return npnized

class MapManager:
    def __init__(self):
        msg_name  = "/move_base_node/local_costmap/costmap"
        self.sub = rospy.Subscriber(msg_name, OccupancyGrid, self.map_callback)
        self.msg = None
        self.arr = None

    def map_callback(self, msg):
        print("rec")
        self.msg = copy.deepcopy(msg)
        info = msg.info
        arr = gridmsg2nparray(msg)
        self.arr = arr

    def show_map(self):
        plt.pcolor(self.arr)

if __name__=='__main__':
    rospy.init_node('wifi_publisher')
    mm = MapManager();
    r = rospy.Rate(10)
    for i in range(20):
        r.sleep()
    mm.show_map()

