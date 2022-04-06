#include "tf/LinearMath/Vector3.h"
#include <ros/ros.h>
#include <tf/transform_listener.h>

tf::StampedTransform get_transform(std::string src, std::string target, tf::TransformListener& listener){
  tf::StampedTransform tf_src_to_target;

  while(true){
    try{
      listener.lookupTransform(target, src, ros::Time(0), tf_src_to_target);
      break;
    }catch (tf::TransformException ex){
      ros::Duration(0.2).sleep();
    }
  }
  return tf_src_to_target;
}

int main(int argc, char** argv){
  ros::init(argc, argv, "my_tf_listener");

  ros::NodeHandle node;

  tf::TransformListener listener;
  const auto tf_foot2base = get_transform("/base_footprint", "/base_link", listener);
  const auto tf_base2torso = get_transform("/base_link", "/torso_lift_link", listener);
  const auto tf_foot2tosro = get_transform("/base_footprint", "/torso_lift_link", listener);
  const auto tf_foot2tosro_computed = tf_foot2base * tf_base2torso;

  const auto origin = tf::Vector3(0.0, 0.0, 0.0);
  std::cout << tf_foot2tosro(origin).getZ() << std::endl;
  std::cout << tf_foot2tosro_computed(origin).getZ() << std::endl;
};
