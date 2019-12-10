using RobotOS
@rosimport std_srvs.srv.Empty
rostypegen()
using .std_srvs.srv 

init_node("a")

function handler(req::EmptyRequest)
  println("recieved!")
  resp = EmptyResponse()
  return resp
end

const srvlisten = Service("test", Empty, handler)
println("start!\n")
spin()

