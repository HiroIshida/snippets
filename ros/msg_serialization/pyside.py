import io
import rospy
from geometry_msgs.msg import Pose

pose = Pose()
pose.position.y = 2.0
pose.orientation.z = 3.0

s = io.BytesIO()
pose.serialize(s)
bindata = s.getvalue()

with open("data.txt", "w") as f:
    f.write(bindata)

