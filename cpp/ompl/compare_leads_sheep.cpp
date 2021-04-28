#include <ompl/base/spaces/ReedsSheppStateSpace.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/State.h>
#include <array>
#include <iostream>

using A3 = std::array<double, 3>;

void set_state(const A3& pose, ompl::base::ScopedState<ompl::base::SE2StateSpace>& s)
{
  s->setX(pose[0]);
  s->setY(pose[1]);
  s->setYaw(pose[2]);
}

double reeds_shepp_cost(const A3& arr_start, const A3& arr_goal){
  ompl::base::StateSpacePtr rspace(new ompl::base::ReedsSheppStateSpace());
  ompl::base::StateSpacePtr space(new ompl::base::SE2StateSpace());
  ompl::base::ScopedState<ompl::base::SE2StateSpace> start(space), goal(space);
  set_state(arr_start, start);
  set_state(arr_goal, goal);
  return rspace->distance(start.get(), goal.get());
}

void alloc_perf(int N){
  for(int i=0; i<N; i++){
    ompl::base::StateSpacePtr rspace(new ompl::base::ReedsSheppStateSpace());
    ompl::base::StateSpacePtr space(new ompl::base::SE2StateSpace());
  }
}

int main(){
  A3 a = {0, 0, 0};
  A3 b = {1, 0, 0};
  A3 c = {0, 1, 0};
  std::cout << "cost to move along x-axis: " << reeds_shepp_cost(a, b) << std::endl; 
  std::cout << "cost to move along y-axis: " << reeds_shepp_cost(a, c) << std::endl; 

  clock_t start = clock();
  alloc_perf(20000);
  clock_t end = clock();
  std::cout << (end - start)/1000.0 << std::endl;  // 57 [ms]
}
