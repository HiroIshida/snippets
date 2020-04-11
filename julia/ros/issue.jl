using RobotOS
@rosimport control_msgs.msg: JointTolerance
#rostypegen()

pkgdps = RobotOS._collectdeps(RobotOS._rospy_imports)
pkglst = RobotOS._order(pkgdps)

code = RobotOS.buildpackage_debug(RobotOS._rospy_imports[pkglst[1]], Main)

open("dump_debug.jl", "w" ) do fp
  write( fp, repr(code))
end
#Main.eval(code)
"""
import .control_msgs.msg: JointTolerance
j = JointTolerance()
println(j)
"""
"""
open("dump_debug.jl", "w" ) do fp
  write( fp, repr(code))
end
"""


