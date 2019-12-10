using RobotOS
@rosimport sample.srv.MyEmpty
rostypegen()
using .sample.srv 

init_node("a")

function handler(req::MyEmptyRequest)
  println("recieved!")
  resp = MyEmptyResponse()
  return resp
end

const srvlisten = Service("test", MyEmpty, handler)
println("start!\n")
spin()

