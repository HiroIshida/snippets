#include <memory>
#include <unordered_map>>
#include <execinfo.h>
#include <cxxabi.h>
#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
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

std::unordered_map<int32_t, std::vector<std::string>> g_stack_trace_cache; 

int32_t djb2_hash(const std::string& str) {
  unsigned long hash = 5381;
  for (char c : str) {
    hash = ((hash << 5) + hash) + c;
  }
  return hash;
}

int32_t djb2_hash(const std::vector<std::string> &strs) {
  // concat all strings
  std::string con;
  for(const auto& s : strs) {
    con += s;
  }
  return djb2_hash(con);
}


std::string demangle(const char* symbol) {
    int status;
    std::unique_ptr<char, void(*)(void*)> demangled(
        abi::__cxa_demangle(symbol, nullptr, nullptr, &status),
        std::free
    );
    return (status == 0) ? demangled.get() : symbol;
}

std::vector<std::string> get_stack_trace() {
    std::vector<std::string> stack_trace;
    void* callstack[128];
    int frames = backtrace(callstack, 128);
    char** symbols = backtrace_symbols(callstack, frames);

    for (int i = 0; i < frames; ++i) {
        Dl_info info;
        if (dladdr(callstack[i], &info) && info.dli_sname) {
            std::string name = demangle(info.dli_sname);
            stack_trace.push_back(name);
        } else {
            stack_trace.push_back(symbols[i]);
        }
    }

    free(symbols);
    return stack_trace;
}


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

  bool project(Eigen::Ref<Eigen::VectorXd> x) const override {
    auto stack_trace = get_stack_trace();
    auto hash_id = djb2_hash(stack_trace);
    if(g_stack_trace_cache.find(hash_id) == g_stack_trace_cache.end()) {
      g_stack_trace_cache[hash_id] = stack_trace;
    }
    return ob::Constraint::project(x);
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
  // show stacktrace
  for(auto& it : g_stack_trace_cache) {
    std::cout << "callstak for Hash: " << it.first << std::endl;
    for(auto& s : it.second) {
      std::cout << "  " << s << std::endl;
    }
  }

  ss->simplifySolution(3);
}
