import time
import rospy
from sensor_msgs.msg import Image, PointCloud2


class Demo:
    # This is simple demo to show the race condition in ROS node
    # NOTE: chatgpt(o1-pro) always lies that rospy is thread-safe or brah brah brah shit
    # just ignore chatgpt.
    busy: bool = True

    def __init__(self):
        image_topic = "/kinect_head/rgb/image_rect_color"
        point_cloud_topic = "/kinect_head/depth_registered/points"
        rospy.Subscriber(image_topic, Image, self.image_callback)
        rospy.Subscriber(point_cloud_topic, PointCloud2, self.point_cloud_callback)
        self.busy = False

    def image_callback(self, msg: Image):
        assert not self.busy, "race condition"
        self.busy = True
        time.sleep(2)
        self.busy = False

    def point_cloud_callback(self, msg: PointCloud2):
        assert not self.busy, "race condition"
        self.busy = True
        time.sleep(1.0)
        self.busy = False


if __name__ == "__main__":
    rospy.init_node("demo")
    node = Demo()
    rospy.spin()
