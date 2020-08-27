# see this : https://answers.ros.org/question/60209/what-is-the-proper-way-to-create-a-header-with-python/
import rospy
import copy 
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header
import tf.transformations as tfs
rospy.init_node("sample")

pub = rospy.Publisher('sample_path', Path, queue_size=1)

rate = rospy.Rate(10) # 10hz
while not rospy.is_shutdown():
    path = Path()

    header = path.header 
    header.frame_id = "base_link"
    header.stamp = rospy.Time.now()

    header_copied = copy.deepcopy(header)

    poses = path.poses
    for i in range(5):
        ps = PoseStamped()
        ps.header = header_copied
        pos = ps.pose.position
        pos.x = float(i) * 0.2
        pos.z = (float(i)**2) * 0.2
        poses.append(ps)

    pub.publish(path)


