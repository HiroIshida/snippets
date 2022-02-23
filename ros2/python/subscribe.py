import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from autoware_planning_msgs.msg import Trajectory

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from math import *

# compare current pose and goal pose

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            Trajectory,
            '/planning/scenario_planning/trajectory',
            self.trajectory_callback,
            10)

        self.subscription  # prevent unused variable warning

        self.goal_pose = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.ego_poes = None
        #self.timer = self.create_timer(0.01, self.on_timer)

    def get_ego(self):
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform('map', 'base_link', now)
        except TransformException as ex:
            print("cannot transform")
            return
        return trans.transform.translation

    def trajectory_callback(self, msg: Trajectory):
        goal_pos = msg.points[0].pose.position
        ego_pos = self.get_ego()
        dist = sqrt((goal_pos.x - ego_pos.x)**2 + (goal_pos.y - ego_pos.y)**2)
        print(dist)


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
