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

contextはここで定義されている. ompl_interface/include/moveit/ompl_interface/model_based_planning_context.h
```
`context->setGoalConstraints`では`goal_constraints_.push_back(kset);`が呼ばれている.
```
ここでgoal_constraints_はこのかたで, 
```
std::vector<kinematic_constraints::KinematicConstraintSetPtr> goal_constraints_;
```
KinematicConstraintはrobot kinematicsを考慮したpose, positionの制約条件.

goalはexplicitに表されるのではなく, samplerとして陰的に表現されている.
```cpp
class ConstrainedGoalSampler : public ompl::base::GoalLazySamples
```

この関数の中で
```cpp
ompl::base::GoalPtr ompl_interface::ModelBasedPlanningContext::constructGoal()
```
次のコードにより, constraint samplerをgoalsに追加していき
```cpp
ob::GoalPtr goal = ob::GoalPtr(new ConstrainedGoalSampler(this, goal_constraint, constraint_sampler));
  goals.push_back(goal);
```
, それを次の関数内で`simple_setup_`にsetする.
```
bool ompl_interface::ModelBasedPlanningContext::setGoalConstraints(...){
    ...
    ompl_simple_setup_->setGoal(goal);
    ...
}
```

## constraintからのsamplingはどうなってる?

### 一様にサンプリングするとめちゃくちゃ時間かかる気がするけど, どうやってサンプリングしてる?
KinematicConstraint 自体にはsamplingの関数はないみたい.
`constrained_sampler.h`にはこんな関数が用意されている.
```
  void sampleUniform(ompl::base::State* state) override;

  /** @brief Sample a state (uniformly) within a certain distance of another state*/
  void sampleUniformNear(ompl::base::State* state, const ompl::base::State* near, const double distance) override;

  /** @brief Sample a state using the specified Gaussian*/
  void sampleGaussian(ompl::base::State* state, const ompl::base::State* mean, const double stdDev) override;
private:
  bool sampleC(ompl::base::State* state);
```
制約ありでconfigurationをsamplingするためには, IKsamplerみたいなものがあって, そこからikをといているらしい.
```
moveit/moveit_core/constraint_samplers場所
bool IKConstraintSampler::sample(moveit::core::RobotState& state, const moveit::core::RobotState& reference_state,
                                 unsigned int max_attempts)
```

### mainのplanningスレッドと並列にsamplingスレッドがあるみたい
https://groups.google.com/g/moveit-users/c/LWrf5E1sQZA?pli=1
では以下のようなやりとりがあり, まず少なくともgoalをsamplingし, 並列でgoalsamplingを回すとった感じらしい.
```
I would like to start solving my planning problem in OMPL from the moment when at least one goal state is found.
Until then, I want my solver (RRT*) to let the priority to the ConstrainedGoalSampler class from MoveIt in order to get a valid goal state.
I know the ConstrainedGoalSampler is called in ompl through a thread created by the GoalLazySamples class. However, I can't figure out the frequency of this call and how I could keep this thread running until a valid state is found.
Thank in advance for your help!

Sonny
```

```
The way goals are set up in MoveIt is such that I think what you want already happens.
Given a goal, a planner is started in one thread and the goal sampling is in another thread. The planner waits until at least one goal sample is available and starts computation. It then adds goal samples to the tree, as the other thread finds them. The goal sampling thread continuously generates samples until a maximum number of attempts or  a maximum number of goal samples.

Ioan
```
