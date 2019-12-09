#!/usr/bin/env julia
using RobotOS
@rosimport std_msgs.msg: String
rostypegen()
import .std_msgs.msg: StringMsg

function main()
    init_node("rosjl_example")
    pub = Publisher{StringMsg}("pts", queue_size=10)
    loop_rate = Rate(5.0)
    while ! is_shutdown()
        str = StringMsg("hoge")
        publish(pub, str)
        rossleep(loop_rate)
    end
end
if ! isinteractive()
    main()
end
