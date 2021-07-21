#include "freespace_planning_algorithms/informed_rrtstar.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

using namespace std;
using namespace rrtstar;

int main(){
  const rrtstar::Pose x_goal{0.4, 0.9, 0.};
  const rrtstar::Pose x_lo{0, 0, -6.28};
  const rrtstar::Pose x_hi{1., 1., +6.28};
  auto lambda = [](const rrtstar::Pose & p) {return true;};
  const auto cspace = rrtstar::CSpace(x_lo, x_hi, 0.2, lambda);

  ifstream test_data("/tmp/reeds_shepp_test_cases.json");
  nlohmann::json js;
  test_data >> js;
  const rrtstar::Pose x_start{0., 0., 0.};
  for(auto& test : js){
    vector<double> vec = test[0];
    vector<vector<double>> res = test[1];
    const rrtstar::Pose x_goal{vec[0], vec[1], vec[2]};
    vector<rrtstar::Pose> pose_seq;
    cspace.sampleWaypoints(x_start, x_goal, 0.02, pose_seq);
    if(pose_seq.size() != res.size()){
      std::cout << "error" << std::endl; 
      return -1;
    }
    for(int i=0; i<res.size(); i++){
      auto & pose = pose_seq[i];
      auto & pose_gs = res[i];
      std::cout << (pose_gs[0] - pose.x) << std::endl; 
      std::cout << (pose_gs[1] - pose.y) << std::endl; 
      std::cout << (pose_gs[2] - pose.yaw) << std::endl; 
    }
  }
}
