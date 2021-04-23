# buildしたコントローラがどのように呼ばれているか?
例えばgazeboでは`pr2_controller_configuration_gazebo`パッケージ内で
```xml
  <!-- Controllers that come up started -->
  <node name="default_controllers_spawner"
        pkg="pr2_controller_manager" type="spawner" output="screen"
        args="--wait-for=/calibrated base_controller base_odometry head_traj_controller laser_tilt_controller torso_controller r_gripper_controller l_gripper_controller r_arm_controller l_arm_controller" />
```
という感じで各種コントローラがロードされている. (plugin lib)

例えば`jsk_pr2_robot`では`pr2_bringup.launch`の中で, 

```xml
  <!-- Default controllers -->
  <include file="$(find pr2_controller_configuration)/pr2_default_controllers.launch" />
```
が呼ばれており, `pr2_robot/pr2_controller_configuration`の中で

```
  <!-- Controllers that come up started -->
  <node name="default_controllers_spawner"
        pkg="pr2_controller_manager" type="spawner" output="screen"
        args="--wait-for=calibrated base_controller base_odometry head_traj_controller laser_tilt_controller torso_controller r_gripper_controller l_gripper_controller r_arm_controller l_arm_controller" />
```
が呼ばれている. 

## pr2_base_controller.cpp
コールバック内で, `cmd_vel_t_` (もとの`cmd_vel_msg`をクランプしたもの)をセットしていて, update()内でその値を補間したものを`cmd_vel_`として保存し, `setJointCommands()`でコマンドを送信している. なお, update関数は周期的によばれている. update関数内では`computeDesiredCasterSteer(dT); computeDesiredWheelSpeeds();` などが呼ばれており, 中で`base_kin_`の中の`wheel_speed_cmd_`や`steer_velocity_desired_`などを更新している. 
```c++
    base_kin_.wheel_[i].wheel_speed_cmd_ = (wheel_point_velocity_projected.linear.x + wheel_caster_steer_component.linear.x) / (base_kin_.wheel_[i].wheel_radius_);

    base_kin_.caster_[i].steer_velocity_desired_ = caster_position_pid_[i].computeCommand(
          error_steer,
          filtered_velocity_[i],
          ros::Duration(dT));
```
制御方法を変更するためには, 結論からいえば, この`computeDesiredCasterSteer`や`computeDesiredWheelSpeeds`などを書き換えればよい. 


### commandCallback関数とsetCommand関数
```c++
void Pr2BaseController::commandCallback(const geometry_msgs::TwistConstPtr& msg)
{
  pthread_mutex_lock(&pr2_base_controller_lock_);
  base_vel_msg_ = *msg;
  this->setCommand(base_vel_msg_);
  pthread_mutex_unlock(&pr2_base_controller_lock_);
}
void Pr2BaseController::setCommand(const geometry_msgs::Twist &cmd_vel)
{
  double vel_mag = sqrt(cmd_vel.linear.x * cmd_vel.linear.x + cmd_vel.linear.y * cmd_vel.linear.y);
  double clamped_vel_mag = filters::clamp(vel_mag,-max_translational_velocity_, max_translational_velocity_);
  if(vel_mag > EPS)
  {
    cmd_vel_t_.linear.x = cmd_vel.linear.x * clamped_vel_mag / vel_mag;
    cmd_vel_t_.linear.y = cmd_vel.linear.y * clamped_vel_mag / vel_mag;
  }
  else
  {
    cmd_vel_t_.linear.x = 0.0;
    cmd_vel_t_.linear.y = 0.0;
  }
  cmd_vel_t_.angular.z = filters::clamp(cmd_vel.angular.z, -max_rotational_velocity_, max_rotational_velocity_);
  cmd_received_timestamp_ = base_kin_.robot_state_->getTime();

  ROS_DEBUG("BaseController:: command received: %f %f %f",cmd_vel.linear.x,cmd_vel.linear.y,cmd_vel.angular.z);
  ROS_DEBUG("BaseController:: command current: %f %f %f", cmd_vel_.linear.x,cmd_vel_.linear.y,cmd_vel_.angular.z);
  ROS_DEBUG("BaseController:: clamped vel: %f", clamped_vel_mag);
  ROS_DEBUG("BaseController:: vel: %f", vel_mag);

  for(int i=0; i < (int) base_kin_.num_wheels_; i++)
  {
    ROS_DEBUG("BaseController:: wheel speed cmd:: %d %f",i,(base_kin_.wheel_[i].direction_multiplier_*base_kin_.wheel_[i].wheel_speed_cmd_));
  }
  for(int i=0; i < (int) base_kin_.num_casters_; i++)
  {
    ROS_DEBUG("BaseController:: caster speed cmd:: %d %f",i,(base_kin_.caster_[i].steer_velocity_desired_));
  }
  new_cmd_available_ = true;
}
```

### update関数
```c++
void Pr2BaseController::update()
{
  ros::Time current_time = base_kin_.robot_state_->getTime();
  double dT = std::min<double>((current_time - last_time_).toSec(), base_kin_.MAX_DT_);

  if(new_cmd_available_)
  {
    if(pthread_mutex_trylock(&pr2_base_controller_lock_) == 0)
    {
      desired_vel_.linear.x = cmd_vel_t_.linear.x;
      desired_vel_.linear.y = cmd_vel_t_.linear.y;
      desired_vel_.angular.z = cmd_vel_t_.angular.z;
      new_cmd_available_ = false;
      pthread_mutex_unlock(&pr2_base_controller_lock_);
    }
  }

  if((current_time - cmd_received_timestamp_).toSec() > timeout_)
  {
    cmd_vel_.linear.x = 0;
    cmd_vel_.linear.y = 0;
    cmd_vel_.angular.z = 0;
  }
  else
    cmd_vel_ = interpolateCommand(cmd_vel_, desired_vel_, max_accel_, dT);

  computeJointCommands(dT);

  setJointCommands();

  updateJointControllers();

  if(publish_state_)
    publishState(current_time);

  last_time_ = current_time;

}
```
