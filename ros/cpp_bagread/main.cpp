#include <iostream>
#include <string>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <nav_msgs/OccupancyGrid.h>

#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

using namespace std;

// http://wiki.ros.org/rosbag/Code%20API
//
int main(){
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

