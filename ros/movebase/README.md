# code reading
move_base.cpp読む.
```cpp
as_ = new MoveBaseActionServer(ros::NodeHandle(), "move_base", boost::bind(&MoveBase::executeCb, this, _1), false);
```
ただし, typedefされてることに注意. 
```cpp
typedef actionlib::SimpleActionServer<move_base_msgs::MoveBaseAction> MoveBaseActionServer;
```
executeCbの中では, `executeCycle(goal)`が呼ばれてて, これが高レベルのメイン処理になる. この中ではエラー検知やrecovery動作の呼び出しなどを行う. 

中レイヤーの処理, plannerそのものは別スレッド`planner_thread_`の中で走っている. 
```cpp
planner_thread_ = new boost::thread(boost::bind(&MoveBase::planThread, this));
```
`planner_thread_`の中で`MoveBase::makePlan`が走っており, さらにその中で`boost::shared_ptr<nav_core::BaseGlobalPlanner> planner_`の`makePlan`が走っている. `makePlan`に失敗した場合は, `recovery_trigger_ = PLANNING_R;` がセットされ, recoveryモードに入る. 
