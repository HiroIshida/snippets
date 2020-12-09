#!/usr/bin/env julia
using RobotOS
@rosimport voxblox_msgs.msg: SignedDistanceField
rostypegen()
import .voxblox_msgs.msg: SignedDistanceField

flag = Bool[true]
global_data = nothing
function callback(msg)
    global global_data = msg
    flag[1] = false
end

try
    isInit
catch
    init_node("subscribe")
    ros_sub = Subscriber("voxblox_node/sd_array", SignedDistanceField, callback, (), queue_size = 10)
    isInit = true
end

println("start")

while(flag[1])
  rossleep(Duration(0.1))
end

msg = global_data
n_points = msg.nx * msg.ny * msg.nz
origin = [msg.origin.x, msg.origin.y, msg.origin.z]
pts = []
idx = 1
for k in 1:msg.nz
    for j in 1:msg.ny
        for i in 1:msg.nx
            if msg.data[idx] < -0.01
                push!(pts, origin + msg.dx * [i, j, k])
            end
            idx += 1
        end
    end
end
pts = hcat(pts...)

using Plots
pyplot()
scatter3d(pts[1, :], pts[2, :], pts[3, :])

