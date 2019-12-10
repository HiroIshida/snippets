using RobotOS
@rosimport sample.srv.SimpleString
rostypegen()
using .sample.srv 

init_node("a")

mutable struct TMP
  val
end
tmp = TMP(nothing)

function handler(req::SimpleStringRequest)
  println("recieved!")
  tmp.val = req
  resp = SimpleStringResponse()
  return resp
end

const srvlisten = Service("test", SimpleString, handler)
println("start!\n")
spin()

