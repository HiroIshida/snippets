#include <nlohmann/json.hpp>
#include <iostream>
#define print(a) std::cout<<a<<std::endl

int main(){
  nlohmann::json group_json = {
    {"torso", {"torso_lift_joint"}},
    {"rarm", {"shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint", "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"}},
    {"head", {"head_pan_joint", "head_tilt_joint"}},
    {"gripper", {"r_gripper_finger_joint", "l_gripper_finger_joint"}}
  };

  auto& j = group_json["rarm"];
  for(auto it = j.begin(); it!=j.end(); it++){
    print(*it);
  }

  std::string s = group_json.dump(4); // adding 4 makes output pretty
  print(s);
}
