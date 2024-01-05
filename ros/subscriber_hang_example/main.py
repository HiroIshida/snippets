import time
import rospy
from rospy.msg import AnyMsg

_global = {"value": None}

def cb1(msg):
    rospy.loginfo("cb1 called")
    while _global["value"] is None:
        time.sleep(0.1)
    rospy.loginfo("global value set")


def cb2(msg):
    rospy.loginfo("cb2 called")
    _global["value"] = msg


if __name__ == "__main__":
    rospy.init_node("test")
    topic_name = "/kinect_head/rgb/image_rect_color"  # whatever
    # changing the order of the subscribers changes the behavior
    # the key to reproduce that cb1 is called before cb2 is to have cb1
    rospy.Subscriber(topic_name, AnyMsg, cb1)
    rospy.Subscriber(topic_name, AnyMsg, cb2)
    rospy.spin()
