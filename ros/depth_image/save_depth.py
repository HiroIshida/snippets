import rospy
import pickle
from sensor_msgs.msg import Image

def callback(_msg):
    rospy.loginfo('rec msg')
    with open('depth.pkl3', 'wb') as f:
        pickle.dump(msg['msg'], f)

rospy.init_node('listener')
sub = rospy.Subscriber('/kinect_head/depth_registered/image', Image, callback)
rospy.spin()
