残念ながらcomponentのデバッグは無理(or簡単には無理)そう. 手作業で単体ノード化してデバッグするしかないかな...

# Suppose, I want to debug `mission_planning_container`
```bash
h-ishida@ishidax1:~$ ros2 component types|grep mission
```
results in
```
mission_planner
  mission_planner::MissionPlannerLanelet2
  mission_planner::GoalPoseVisualizer
```

```bash
ps -aux|grep mission_planning_container
```
results in
```
h-ishida@ishidax1:~$ ps -aux|grep mission_planning_container
h-ishida  128680  1.2  0.1 638720 36756 pts/0    Sl+  03:51   0:03 /opt/ros/galactic/lib/rclcpp_components/component_container --ros-args -r __node:=mission_planning_container -r __ns:=/planning/mission_planning --params-file /tmp/launch_params_6qv81r2m
h-ishida  133952  0.0  0.0  18820  3028 pts/2    S+   03:55   0:00 grep --color=auto mission_planning_container
```
