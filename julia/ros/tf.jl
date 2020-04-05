using RobotOS
using PyCall

const __tf__ = PyCall.PyNULL()
copy!(__tf__, pyimport("tf"))

init_node("test")
a = __tf__.TransformListener()
a.lookupTransform("base_link", "base_link", RobotOS.__rospy__.Time())


