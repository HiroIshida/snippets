#include <iostream>
#include <string>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/Pose.h>

#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

using namespace std;

// http://wiki.ros.org/rosbag/Code%20API
void write_and_read(){
  rosbag::Bag bag;
  bag.open("tmp.bag", rosbag::bagmode::Write);
  geometry_msgs::Pose pose;
  pose.position.x = 1.0;
  pose.position.y = 2.0;
  pose.position.z = 3.0;
  pose.orientation.z = 1.0;
  ros::Time dummy;
  dummy.sec = 1.0;
  bag.write("mypose", dummy, pose);
  bag.close();

  bag.open("tmp.bag", rosbag::bagmode::Read);
  vector<string> topics;
  topics.push_back("mypose");
  rosbag::View view(bag, rosbag::TopicQuery(topics));

  foreach(rosbag::MessageInstance const m, view){
    auto s = m.instantiate<geometry_msgs::Pose>();
    std::cout << s->position.x << std::endl; 
    std::cout << s->position.y << std::endl; 
    std::cout << s->position.z << std::endl; 
  }
  bag.close();
}

int main(){
  write_and_read();
  rosbag::Bag bag;
  bag.open("../costmap.bag", rosbag::bagmode::Read);
  std::cout << "bag open!" << std::endl; 
  vector<string> topics;
  topics.push_back("/planning/scenario_planning/parking/costmap_generator/occupancy_grid");
  rosbag::View view(bag, rosbag::TopicQuery(topics));

  foreach(rosbag::MessageInstance const m, view){
    nav_msgs::OccupancyGrid::ConstPtr s = m.instantiate<nav_msgs::OccupancyGrid>();
    std::cout << s->header << std::endl; 
  }
  bag.close();
}

