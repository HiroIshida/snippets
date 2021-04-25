import time
import rospy
from geometry_msgs.msg import Twist
from pr2_mechanism_controllers.msg import BaseDirectCommand

pub = rospy.Publisher("base_controller/direct_command", BaseDirectCommand, queue_size=1)
rospy.init_node('tmp', anonymous=True)

def create_cmd():
    cmd = BaseDirectCommand()
    cmd.wheel_vels = [0.0 for _ in range(8)]
    cmd.caster_vels = [0.0 for _ in range(4)]
    return cmd

def straight(vel_straight, cmd):
    cmd.wheel_vels = [0.0 for _ in range(8)]
    for i in range(8):
        cmd.wheel_vels[i] = vel_straight

def rotate(vel_rotate, cmd):
    for i in range(4):
        cmd.caster_vels[i] = vel_rotate

for i in range(20):
    cmd = create_cmd()
    rotate(20, cmd)
    pub.publish(cmd)
    time.sleep(0.2)
print("finish rotating")

for i in range(20):
    cmd = create_cmd()
    straight(20, cmd)
    pub.publish(cmd)
    time.sleep(0.2)
print("finish straight")

