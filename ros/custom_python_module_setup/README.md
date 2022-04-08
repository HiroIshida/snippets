## what is this?
A drawback of `catkin_python_setup()` of `catkin` is it's append the path in the "run time". This causes difficulty in modern way of python development where static type checking and auto completion are heavily used.

Instead, `cmake/setup_python_as_submodule.cmake` create a symbolic link of all modules under `package_path/python` directory. Therefore, one can take advantage all the static checking tool for developing a python module inside catkin package.

### Info of original implementation
In the runtime, path is inserted. https://github.com/ros/catkin/blob/be490b217e66f0a175b48eaa70062976eb67ffd6/cmake/templates/__init__.py.in#L11

