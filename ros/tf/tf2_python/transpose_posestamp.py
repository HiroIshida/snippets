# ref: https://answers.ros.org/question/222306/transform-a-pose-to-another-frame-with-tf2-in-python/
import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped

# please connect to PR2 beforehand
rospy.init_node("tmp", anonymous=True)
tf_buffer = tf2_ros.Buffer(rospy.Duration(1200.0)) #tf buffer length
tf_listener = tf2_ros.TransformListener(tf_buffer)

transform = tf_buffer.lookup_transform(
        "base_link", "base_footprint",
        rospy.Time(0), # get the tf at first available time
        rospy.Duration(1.0)) # wait for 1 second

ps = PoseStamped()
ps.header.frame_id = "base_link"
trans = ps.pose.position
rot = ps.pose.orientation
rot.w = 1.0

pose_transformed = tf2_geometry_msgs.do_transform_pose(ps, transform)
print(pose_transformed)

