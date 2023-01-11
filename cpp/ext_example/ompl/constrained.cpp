#include <memory>
#include <ompl/base/Constraint.h>
#include <ompl/base/Planner.h>
#include <ompl/base/spaces/RealVectorBounds.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/spaces/constraint/ProjectedStateSpace.h>
#include <ompl/base/spaces/constraint/ConstrainedStateSpace.h>
#include <ompl/base/ConstrainedSpaceInformation.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/planners/rrt/RRT.h>
#include <stdexcept>

namespace ob = ompl::base;
namespace og = ompl::geometric;


class SphereConstraint : public ob::Constraint
{
public:
    SphereConstraint() : ob::Constraint(3, 1) {}

    void function(const Eigen::Ref<const Eigen::VectorXd> &x, Eigen::Ref<Eigen::VectorXd> out) const override
    {
        out[0] = x.norm() - 1;
    }

    void jacobian(const Eigen::Ref<const Eigen::VectorXd> &x, Eigen::Ref<Eigen::MatrixXd> out) const override
    {
        out = x.transpose().normalized();
    }
};

bool is_valid(const ob::State *state){
  return true;
}

int main(){
  auto space = std::make_shared<ob::RealVectorStateSpace>(3);

  ob::RealVectorBounds bounds(3);
  bounds.setLow(-2);
  bounds.setHigh(2);
  space->setBounds(bounds);

  // Create a shared pointer to our constraint.
  auto constraint = std::make_shared<SphereConstraint>();

  const auto css = std::make_shared<ob::ProjectedStateSpace>(space, constraint);
  const auto csi = std::make_shared<ob::ConstrainedSpaceInformation>(css);

  Eigen::VectorXd start(3), goal(3);
  start << 1, 0, 0;
  goal << -1, 0, 0;

  ob::ScopedState<> sstart(css);
  ob::ScopedState<> sgoal(css);

  sstart->as<ob::ConstrainedStateSpace::StateType>()->copy(start);
  sgoal->as<ob::ConstrainedStateSpace::StateType>()->copy(goal);

  const auto ss = std::make_shared<og::SimpleSetup>(csi);
  ss->setStateValidityChecker([](const ob::State * state){return true;});
  ss->setStartAndGoalStates(sstart, sgoal);

  const auto algo = std::make_shared<og::RRT>(csi);
  ss->setPlanner(std::static_pointer_cast<ob::Planner>(algo));
  ss->setup();
  ss->solve(10);

  ss->simplifySolution(3);
  auto simplePath = ss->getSolutionPath();
  std::cout << "length: " << simplePath.length() << std::endl;
  const auto p = ss->getSolutionPath().as<og::PathGeometric>();
  const auto& states = p->getStates();
  for (const auto& state : states) {
    std::vector<double> reals;
    const auto ss = state->as<ompl::base::ConstrainedStateSpace::StateType>()->getState()->as<ob::RealVectorStateSpace::StateType>();
    space->copyToReals(reals, ss);
    std::cout << reals.at(0) << ", " << reals.at(1) << ", " << reals.at(2) << std::endl;
  }
}
