#include <ompl/base/spaces/ReedsSheppStateSpace.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/State.h>

#include <array>
#include <iostream>

using namespace std;
using A3 = std::array<double, 3>;

void set_state(A3 pose, ompl::base::ScopedState<ompl::base::SE2StateSpace>& s)
{
  s->setX(pose[0]);
  s->setY(pose[1]);
  s->setYaw(pose[2]);
}

double reeds_shepp_cost(A3 arr_start, A3 arr_goal){
  ompl::base::StateSpacePtr rspace(new ompl::base::ReedsSheppStateSpace());
  ompl::base::StateSpacePtr space(new ompl::base::SE2StateSpace());
  ompl::base::ScopedState<ompl::base::SE2StateSpace> start(space), goal(space);
  set_state(arr_start, start);
  set_state(arr_goal, goal);
  return rspace->distance(start.get(), goal.get());
}

int main(){
  A3 a = {0, 0, 0};
  A3 b = {1, 1, 1};
  std::cout << reeds_shepp_cost(a, b) << std::endl; 
}
