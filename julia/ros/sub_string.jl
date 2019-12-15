#!/usr/bin/env julia
using RobotOS
@rosimport std_msgs.msg: String
rostypegen()
import .std_msgs.msg: StringMsg

flag = Bool[true]
function callback(msg)
  println("aa")
  flag[1] = false
end

init_node("subscribe")
const ros_sub = Subscriber("test", StringMsg, callback, (), queue_size = 10)

while(flag[1])
  rossleep(Duration(0.1))
end
