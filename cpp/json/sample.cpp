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

  print("=--------------------------");
  for(auto it = group_json.begin(); it!=group_json.end(); it++){
    print(it.key() << " : " << it.value());
  }

  print("=----conversion to std vector. many stl type is supported");
  std::vector<std::string> strvec;
  for(auto& group : group_json.items()){
    print(group.key() << " : ");
     strvec = group.value().get<std::vector<std::string>>();
      for(std::string& s : strvec){
        print(s);
      }
  }

  print("=----dump");
  std::string s = group_json.dump(4); // adding 4 makes output pretty
  print(s);
}
