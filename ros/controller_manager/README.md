これをみる. http://wiki.ros.org/controller_manager この中で`controller_manager`を`pr2_controller_manager`に変更する.
```bash
rosrun pr2_controller_manager pr2_controller_manager list
```

unload する方法. 
```bash
# How to unload the controller
rosrun pr2_controller_manager pr2_controller_manager stop base_controller
rosrun pr2_controller_manager pr2_controller_manager unload base_controller

rosrun pr2_controller_manager pr2_controller_manager load base_controller
rosrun pr2_controller_manager pr2_controller_manager start base_controller
```

自前のコントローラを使うとき, `base_controller`に関するrosparamをオーバーライトすればよい. 
```
rosparam delete base_controller
rosparam load $(rospack find pr2_controller_configuration)/pr2_base_controller.yaml
```


