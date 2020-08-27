# see as for matplotlib to np
# https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array

# see as for np to cv
# https://stackoverflow.com/questions/42603161/convert-an-image-shown-in-python-into-an-opencv-image

from sensor_msgs.msg import Image
import rospy
import copy 
import matplotlib.pyplot as plt
import scipy.interpolate  
import numpy as np

import cv2
from cv_bridge import CvBridge

fig, ax = plt.subplots()
xlin = np.linspace(0.0, 2.0, 200)
ylin = np.linspace(0.0, 1.0, 200)
X, Y = np.meshgrid(xlin, ylin)
Z = X**2 + Y**2
X, Y = np.meshgrid(xlin, ylin)
ax.contourf(X, Y, Z)
fig.canvas.draw()

data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
img = cv2.cvtColor(data,cv2.COLOR_RGB2BGR)

bridge = CvBridge()
msg = bridge.cv2_to_imgmsg(img, 'bgr8')

ros = True
if ros:
    rospy.init_node("matplotlib_image")
    pub = rospy.Publisher('debug_image', Image, queue_size=1)
    while not rospy.is_shutdown():
        pub.publish(msg)
        print("publishing")

