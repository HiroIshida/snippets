import math
import rclpy
from rclpy.node import Node

from std_msgs.msg import Header
from geometry_msgs.msg import Pose                                                                                                                                                    
from geometry_msgs.msg import PoseWithCovarianceStamped                                                                                                                                                    


class AutoStarter(Node):

    def __init__(self):
        super().__init__('auto_starter')
        self.publisher_ = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

    def run(self, pos, ori):
        init_msg = PoseWithCovarianceStamped()
        init_msg.header.stamp = self.get_clock().now().to_msg()
        init_msg.header.frame_id = "map"

        for attr, p in zip(['x', 'y', 'z'], pos):
            setattr(init_msg.pose.pose.position, attr, p)
        for attr, p in zip(['x', 'y', 'z', 'w'], ori):
            setattr(init_msg.pose.pose.orientation, attr, p)
        # set covariance to the same as autoware rviz
        init_msg.pose.covariance[0] = 0.25
        init_msg.pose.covariance[7] = 0.25
        init_msg.pose.covariance[-1] = (15 * math.pi / 180) ** 2

rclpy.init()
ast = AutoStarter()

pos = [3713.895, 73750.87, 0.0]
ori = [0.0, 0.0, 0.23146010852100476, 0.9728443956581364]
ast.run(pos, ori)
