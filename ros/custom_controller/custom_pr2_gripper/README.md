## pr2 controllerがロードされるまで
pr2のコントローラには`r_gripper_controller`というものがある.

`pr2_robot/pr2_controller_configuration/pr2_gripper_controllers.yaml`をみてみると以下のようなyamlfileがあり, 
```yaml
r_gripper_controller:
  type: pr2_mechanism_controllers/Pr2GripperController
  joint: r_gripper_joint
  pid: &gripper_position_gains
    p: 10000.0
    d: 1000.0
```

`pr2_robot/pr2_controller_configuration/pr2_default_controllers.launch`内で以下のように上記のyamlファイルで設定したコントローラをloadする. 
```xml
<!-- Controllers that come up started -->
<node name="default_controllers_spawner"
    pkg="pr2_controller_manager" type="spawner" output="screen"
    args="--wait-for=calibrated base_controller base_odometry head_traj_controller laser_tilt_controller torso_controller r_gripper_controller l_gripper_controller r_arm_controller l_arm_controller" />
```
`pr2_mechanism/pr2_controller_manager/scripts/spawner`を覗いてみると, こんな感じでcontrollerをloadしてる.
```python
load_controller = rospy.ServiceProxy('pr2_controller_manager/load_controller', loadcontroller)
```
なおLoadcontrollerは`from pr2_mechanism_msgs.srv import *`からきてる. 

## gripper actionのパイプライン
client側が`Pr2GripperCommandAction`を送る. それを`pr2_controllers/pr2_gripper_action/src/pr2_gripper_action.cpp`が受け取って処理したあとに`pr2_controllers_msgs::Pr2GripperCommand`の型のコマンドを送っている. たぶんこのサーバノードでは大したことはしていない.
```
[pr2_controllers_msgs/Pr2GripperCommand]:
float64 position
float64 max_effort
```
そして, このコマンドを受け取った`pr2_controllers/pr2_mechanism_controllers/src/pr2_gripper_controller.cpp`がハードウェアにコマンドを送っている. ハードウェアとのやりとりは`command_box_`を介してなされるので, 以下のコードを読む限りソフトウェア側でいじれるのはpositionが限界のようです. 
```cpp
using namespace pr2_controllers_msgs;
Pr2GripperCommandPtr c(new Pr2GripperCommand);
c->position = joint_state_->position_;
c->max_effort = 0.0;
command_box_.set(c);
```

