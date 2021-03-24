#include "geometry_msgs/Pose.h"
#include <fstream>
#include <iostream>
namespace ser = ros::serialization;

int main(){
  geometry_msgs::Pose value;
  uint32_t serial_size = ros::serialization::serializationLength(value);

  std::ifstream in("../data.txt");
  std::string contents((std::istreambuf_iterator<char>(in)), 
  std::istreambuf_iterator<char>());
  std::cout << contents << std::endl; 

  std::cout << contents.c_str() << std::endl;
  int n = contents.size() + 1;

  auto buffer = new uint8_t[n];
  for(int i=0; i<n; i++){
    buffer[i] = contents.c_str()[i];
  }
  ser::IStream stream(buffer, serial_size);
  ser::deserialize(stream, value);
  std::cout << value << std::endl; 
}
