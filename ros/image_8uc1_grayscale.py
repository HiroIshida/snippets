from sensor_msgs.msg import Image
import numpy as np
import rospy


img = Image()
img.height = 40
img.width = 80
img.encoding = '8UC1'
img.is_bigendian = 0
img.step = img.width * 1

data = np.zeros((img.height, img.width)).astype(np.uint8)
data[10:30, 10:30] = 254
data[20:35, 20:60] = 100

img.data = data.flatten().tolist()

rospy.init_node('test_image')
pub = rospy.Publisher('dummy_img', Image, queue_size=10)

rate = rospy.Rate(5)
while not rospy.is_shutdown():
    rospy.loginfo('pub')
    pub.publish(img)
    rate.sleep()
