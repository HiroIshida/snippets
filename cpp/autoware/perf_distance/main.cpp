#include <iostream>
#include "autoware_utils/autoware_utils.h"
#include <geometry_msgs/Pose.h>
//using namespace geometry_msgs;

double my_custom_dist(geometry_msgs::Pose& pose0, geometry_msgs::Pose& pose1){
  auto& p1 = pose0.position;
  auto& p2 = pose1.position;
  return std::hypot(p1.x - p2.x, p1.y - p2.y);
}

int main(){
  int N = 100000000;
  geometry_msgs::Pose pose0;
  pose0.position.x = 0.001;
  geometry_msgs::Pose pose1;

  {
    clock_t start = clock();
    double sum = 0;
    for(int i=0; i<N; i++){
      sum += autoware_utils::calcDistance2d(pose0, pose1);
    }
    std::cout << sum << std::endl; 
    std::cout << clock() - start << std::endl; 
  }

  {
    clock_t start = clock();
    double sum = 0;
    for(int i=0; i<N; i++){
      sum += my_custom_dist(pose0, pose1);
    }
    std::cout << sum << std::endl; 
    std::cout << clock() - start << std::endl; 
  }
}
