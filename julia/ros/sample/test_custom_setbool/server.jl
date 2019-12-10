using RobotOS
@rosimport sample.srv.MySetBool
rostypegen()
using .sample.srv 

init_node("a")

function handler(req::MySetBoolRequest)
  println("recieved!")
  resp = MySetBoolResponse()
  return resp
end

const srvlisten = Service("test", MySetBool, handler)
println("start!\n")
spin()

