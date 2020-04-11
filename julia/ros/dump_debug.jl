:($(Expr(:toplevel, :(module control_msgs
  #= /home/anne/documents/julia/RobotOS.jl/src/gentypes.jl:323 =#
  #= /home/anne/documents/julia/RobotOS.jl/src/gentypes.jl:323 =#
  module msg
  #= /home/anne/documents/julia/RobotOS.jl/src/gentypes.jl:331 =#
  #= /home/anne/documents/julia/RobotOS.jl/src/gentypes.jl:331 =#
  begin
      #= /home/anne/documents/julia/RobotOS.jl/src/gentypes.jl:367 =#
      using PyCall
      #= /home/anne/documents/julia/RobotOS.jl/src/gentypes.jl:368 =#
      import Base: convert, getproperty
      #= /home/anne/documents/julia/RobotOS.jl/src/gentypes.jl:369 =#
      import RobotOS
      #= /home/anne/documents/julia/RobotOS.jl/src/gentypes.jl:370 =#
      import RobotOS.Time
      #= /home/anne/documents/julia/RobotOS.jl/src/gentypes.jl:371 =#
      import RobotOS.Duration
      #= /home/anne/documents/julia/RobotOS.jl/src/gentypes.jl:372 =#
      import RobotOS._typedefault
      #= /home/anne/documents/julia/RobotOS.jl/src/gentypes.jl:373 =#
      import RobotOS._typerepr
  end
  import RobotOS.AbstractMsg
  export JointTolerance
  mutable struct JointTolerance <: AbstractMsg
      #= /home/anne/documents/julia/RobotOS.jl/src/gentypes.jl:498 =#
      name::String
      position::Float64
      velocity::Float64
      acceleration::Float64
  end
  function JointTolerance()
      #= /home/anne/documents/julia/RobotOS.jl/src/gentypes.jl:504 =#
      JointTolerance("", 0.0, 0.0, 0.0)
  end
  function convert(::Type{PyObject}, o::JointTolerance)
      #= /home/anne/documents/julia/RobotOS.jl/src/gentypes.jl:513 =#
      py = pycall(RobotOS._rospy_objects["control_msgs/JointTolerance"], PyObject)
      #= /home/anne/documents/julia/RobotOS.jl/src/gentypes.jl:515 =#
      py.:("name") = convert(PyObject, o.name)
      py.:("position") = convert(PyObject, o.position)
      py.:("velocity") = convert(PyObject, o.velocity)
      py.:("acceleration") = convert(PyObject, o.acceleration)
      py
  end
  function convert(jlt::Type{JointTolerance}, o::PyObject)
      #= /home/anne/documents/julia/RobotOS.jl/src/gentypes.jl:521 =#
      if convert(String, o.:("_type")) != _typerepr(jlt)
          #= /home/anne/documents/julia/RobotOS.jl/src/gentypes.jl:522 =#
          throw(InexactError(:convert, JointTolerance, o))
      end
      #= /home/anne/documents/julia/RobotOS.jl/src/gentypes.jl:524 =#
      jl = JointTolerance()
      #= /home/anne/documents/julia/RobotOS.jl/src/gentypes.jl:526 =#
      jl.name = convert(String, o.:("name"))
      jl.position = convert(Float64, o.:("position"))
      jl.velocity = convert(Float64, o.:("velocity"))
      jl.acceleration = convert(Float64, o.:("acceleration"))
      jl
  end
  function getproperty(::Type{JointTolerance}, s::Symbol)
      #= /home/anne/documents/julia/RobotOS.jl/src/gentypes.jl:532 =#
      try
          #= /home/anne/documents/julia/RobotOS.jl/src/gentypes.jl:532 =#
          getproperty(RobotOS._rospy_objects["control_msgs/JointTolerance"], s)
      catch ex
          #= /home/anne/documents/julia/RobotOS.jl/src/gentypes.jl:534 =#
          ex isa KeyError || rethrow(ex)
          #= /home/anne/documents/julia/RobotOS.jl/src/gentypes.jl:535 =#
          try
              #= /home/anne/documents/julia/RobotOS.jl/src/gentypes.jl:535 =#
              getfield(JointTolerance, s)
          catch ex2
              #= /home/anne/documents/julia/RobotOS.jl/src/gentypes.jl:537 =#
              startswith(ex2.msg, "type DataType has no field") || rethrow(ex2)
              #= /home/anne/documents/julia/RobotOS.jl/src/gentypes.jl:538 =#
              error("Message type '" * "JointTolerance" * "' has no property '$(s)'.")
          end
      end
  end
  _typerepr(::Type{JointTolerance}) = begin
          #= /home/anne/documents/julia/RobotOS.jl/src/gentypes.jl:550 =#
          "control_msgs/JointTolerance"
      end
  end
  import RobotOS.@rosimport
  function __init__()
      #= /home/anne/documents/julia/RobotOS.jl/src/gentypes.jl:324 =#
      #= /home/anne/documents/julia/RobotOS.jl/src/gentypes.jl:338 =# @rosimport control_msgs.msg:JointTolerance
  end
  end))))