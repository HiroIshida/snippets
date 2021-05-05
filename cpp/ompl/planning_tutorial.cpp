#include <ompl/base/spaces/RealVectorBounds.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/State.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <array>
#include <iostream>
namespace ob = ompl::base;
namespace og = ompl::geometric;

void set_state(ob::ScopedState<ompl::base::SE2StateSpace>& state, double x, double y, double yaw){
  state->setX(x);
  state->setY(y);
  state->setYaw(yaw);
}

int main(){
  auto space(std::make_shared<ob::SE2StateSpace>());
  ob::RealVectorBounds bounds(2);
  bounds.setLow(0.);
  bounds.setHigh(1.);
  space->setBounds(bounds);

  auto si(std::make_shared<ob::SpaceInformation>(space));
  auto pdef(std::make_shared<ob::ProblemDefinition>(si));
  auto isStateValid = [&](const ob::State *state){
    auto s = state->as<ob::SE2StateSpace::StateType>();
    double x = (s->getX() - 0.5);
    double y = (s->getY() - 0.5);
    double r = 0.35;
    bool is_valid = ((x*x + y*y) > r*r);
    return is_valid;
  };

  si->setStateValidityChecker(isStateValid);

  ob::ScopedState<ompl::base::SE2StateSpace> start(space), goal(space);
  set_state(start, 0.1, 0.1, 0.);
  set_state(goal, 0.9, 0.9, 0.);

  pdef->setStartAndGoalStates(start, goal);

  auto planner(std::make_shared<og::RRTConnect>(si));
  planner->setProblemDefinition(pdef);
  planner->setup();

  ob::PlannerStatus solved = planner->ob::Planner::solve(10.0);

  if (solved)
  {
      // get the goal representation from the problem definition (not the same as the goal state)
      // and inquire about the found path
      ob::PathPtr path = pdef->getSolutionPath();
      std::cout << "Found solution:" << std::endl;

      // print the path to screen
      path->print(std::cout);
  }
}
