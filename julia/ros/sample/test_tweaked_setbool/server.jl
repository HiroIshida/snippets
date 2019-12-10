using RobotOS
@rosimport sample.srv.TweakedSetBool
rostypegen()
using .sample.srv 

init_node("a")

function handler(req::TweakedSetBoolRequest)
  println("recieved!")
  resp = TweakedSetBoolResponse()
  return resp
end

const srvlisten = Service("test", TweakedSetBool, handler)
println("start!\n")
spin()

