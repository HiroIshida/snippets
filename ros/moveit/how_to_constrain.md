## いかにしてpose_constraintが設定されているか

move_group_interface.cpp

`class MoveGroupInterface::MoveGroupInterfaceImpl`の`pose_targets_` にいれている.
```
bool setPoseTargets(const std::vector<geometry_msgs::PoseStamped>& poses, const std::string& end_effector_link)
```

このあたりでしか読み込んでなさそう.
```c++
const geometry_msgs::PoseStamped& getPoseTarget(const std::string& end_effector_link) const
const std::vector<geometry_msgs::PoseStamped>& getPoseTargets(const std::string& end_effector_link) const


void constructMotionPlanRequest(moveit_msgs::MotionPlanRequest& request) const  // これが一番大事そう
```
`request.goal_constraints`に放り込んでいる.

```cpp
  void constructMotionPlanRequest(moveit_msgs::MotionPlanRequest& request) const
  {
    request.group_name = opt_.group_name_;
    request.num_planning_attempts = num_planning_attempts_;
    request.max_velocity_scaling_factor = max_velocity_scaling_factor_;
    request.max_acceleration_scaling_factor = max_acceleration_scaling_factor_;
    request.cartesian_speed_limited_link = cartesian_speed_limited_link_;
    request.max_cartesian_speed = max_cartesian_speed_;
    request.allowed_planning_time = allowed_planning_time_;
    request.pipeline_id = planning_pipeline_id_;
    request.planner_id = planner_id_;
    request.workspace_parameters = workspace_parameters_;

    if (considered_start_state_)
      moveit::core::robotStateToRobotStateMsg(*considered_start_state_, request.start_state);
    else
      request.start_state.is_diff = true;

    if (active_target_ == JOINT)
    {
      request.goal_constraints.resize(1);
      request.goal_constraints[0] = kinematic_constraints::constructGoalConstraints(
          getTargetRobotState(), joint_model_group_, goal_joint_tolerance_);
    }
    else if (active_target_ == POSE || active_target_ == POSITION || active_target_ == ORIENTATION)
    {
      // find out how many goals are specified
      std::size_t goal_count = 0;
      for (const auto& pose_target : pose_targets_)
        goal_count = std::max(goal_count, pose_target.second.size());

      // start filling the goals;
      // each end effector has a number of possible poses (K) as valid goals
      // but there could be multiple end effectors specified, so we want each end effector
      // to reach the goal that corresponds to the goals of the other end effectors
      request.goal_constraints.resize(goal_count);

      for (const auto& pose_target : pose_targets_)
      {
        for (std::size_t i = 0; i < pose_target.second.size(); ++i)
        {
          moveit_msgs::Constraints c = kinematic_constraints::constructGoalConstraints(
              pose_target.first, pose_target.second[i], goal_position_tolerance_, goal_orientation_tolerance_);
          if (active_target_ == ORIENTATION)
            c.position_constraints.clear();
          if (active_target_ == POSITION)
            c.orientation_constraints.clear();
          request.goal_constraints[i] = kinematic_constraints::mergeConstraints(request.goal_constraints[i], c);
        }
      }
    }
    else
      ROS_ERROR_NAMED(LOGNAME, "Unable to construct MotionPlanRequest representation");

    if (path_constraints_)
      request.path_constraints = *path_constraints_;
    if (trajectory_constraints_)
      request.trajectory_constraints = *trajectory_constraints_;
  }
```

## どうやってOMPLで読んでいるか.

planning_context_manager.cpp

```c++
ompl_interface::ModelBasedPlanningContextPtr ompl_interface::PlanningContextManager::getPlanningContext(
    const planning_scene::PlanningSceneConstPtr& planning_scene, const moveit_msgs::MotionPlanRequest& req,
    moveit_msgs::MoveItErrorCodes& error_code, const ros::NodeHandle& nh, bool use_constraints_approximation) const
{
...
if (!context->setGoalConstraints(req.goal_constraints, req.path_constraints, &error_code))
...
}
```

ただしcontextはここで定義されている.
ompl_interface/include/moveit/ompl_interface/model_based_planning_context.h
```
`context->setGoalConstraints`では`goal_constraints_.push_back(kset);`が呼ばれている.
ただし
```
std::vector<kinematic_constraints::KinematicConstraintSetPtr> goal_constraints_;
```
