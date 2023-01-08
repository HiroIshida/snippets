#include <eigen3/Eigen/Core>
#include <ompl/base/Planner.h>
#include <ompl/base/State.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/planners/experience/LightningRetrieveRepair.h>
#include <ompl/tools/lightning/LightningDB.h>

#include <chrono>
#include <memory>
#include <thread>

namespace ob = ompl::base;
namespace og = ompl::geometric;
namespace ot = ompl::tools;
using namespace std::chrono;


og::SimpleSetupPtr create_setup(){
  const auto space(std::make_shared<ob::RealVectorStateSpace>());
  space->addDimension(0.0, 1.0);
  space->addDimension(0.0, 1.0);
  const auto si = std::make_shared<ob::SpaceInformation>(space);

  const auto isValid = [](const ob::State* s){
    const auto& rs = s->as<ob::RealVectorStateSpace::StateType>();
    const auto x = (rs->values[0] - 0.5);
    const auto y = (rs->values[1] - 0.5);
    const double r = 0.43;
    // NOTE: to compare performance, the sleep is injected intentionaly
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    return (x * x + y * y) > (r * r);
  };

  const auto setup = std::make_shared<og::SimpleSetup>(si);
  setup->setStateValidityChecker(isValid);
  return setup;
}

ot::LightningDBPtr create_database(og::SimpleSetupPtr setup, const int n_data, double& time_per_problem){
  const auto si = setup->getSpaceInformation();
  const auto algo = std::make_shared<og::RRTConnect>(si);
  setup->setPlanner(algo);

  // create many example data

  auto database = std::make_shared<ot::LightningDB>(si->getStateSpace());

  double time_sum_rrt = 0.0;
  for(size_t i = 0; i < n_data; ++i){
    setup->clear();
    const auto valid_sampler = setup->getSpaceInformation()->allocValidStateSampler();
    ob::ScopedState<> start(setup->getStateSpace());
    valid_sampler->sample(start.get());
    ob::ScopedState<> goal(setup->getStateSpace());
    valid_sampler->sample(goal.get());
    setup->setStartAndGoalStates(start, goal);
    const auto result = setup->solve(1.0);
    const bool solved = result && result != ob::PlannerStatus::APPROXIMATE_SOLUTION;
    if(solved){
      auto p = setup->getSolutionPath().as<og::PathGeometric>();
      double insertion_time;
      database->addPath(*p, insertion_time);
    }
    time_sum_rrt += setup->getLastPlanComputationTime();
  }
  time_per_problem = time_sum_rrt / (double)n_data;
  return database;
}

int main(){
  const auto setup = create_setup();
  const auto si = setup->getSpaceInformation();

  double rrt_time;
  const auto database = create_database(setup, 100, rrt_time);

  auto repair_planner = std::make_shared<og::LightningRetrieveRepair>(si, database);
  //const auto tmp = std::make_shared<og::RRTConnect>(si);
  // const ob::PlannerPtr std::static_pointer_cast<ob::Planner>(std::make_shared<og::RRTConnect>(si));

  const auto algo = std::make_shared<og::RRTConnect>(si);
  const auto p = std::static_pointer_cast<ob::Planner>(algo);

  // valid sampler
  const auto valid_sampler = setup->getSpaceInformation()->allocValidStateSampler();
  ob::ScopedState<> start(setup->getStateSpace());
  valid_sampler->sample(start.get());
  ob::ScopedState<> goal(setup->getStateSpace());
  valid_sampler->sample(goal.get());

  repair_planner->setRepairPlanner(p); 

  ob::ProblemDefinitionPtr pdef(new ob::ProblemDefinition(si));
  pdef->setStartAndGoalStates(start.get(), goal.get());
  repair_planner->setProblemDefinition(pdef);
  const std::function<bool()> fn = []() { return false; };
  repair_planner->solve(fn);
}
