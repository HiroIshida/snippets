#include <Eigen/Core>
#include <iostream>
#include <memory>
#include <vector>
#include <chrono>

using namespace Eigen;
using namespace std;

struct Node{
  Node(std::shared_ptr<Node> _parent, Matrix3d _mat) : 
    parent(std::move(_parent)), mat(_mat) {}

  std::shared_ptr<Node> parent;
  Matrix3d mat;
};

Matrix3d recursion(std::shared_ptr<Node> node){
  if(node){
    return recursion(node->parent) * node->mat;
  }else{
    return Eigen::Matrix3d::Zero();
  }
}

Matrix3d iteration(std::shared_ptr<Node> node) {
    if (!node) {
        return Eigen::Matrix3d::Zero();
    }

    Matrix3d result = node->mat;
    node = node->parent;

    while (node) {
        result = node->mat * result;
        node = std::move(node->parent);
    }
    return result;
}

int main(){
  // create data
  std::vector<std::shared_ptr<Node>> tree;
  auto node_parent = std::make_shared<Node>(nullptr, Eigen::Matrix3d::Random());
  tree.push_back(node_parent);
  for(size_t i = 0; i < 20; ++i) { 
    auto child = std::make_shared<Node>(tree.back(), Eigen::Matrix3d::Random());
    tree.push_back(child);
  }

  size_t N_test = 10000000;

  {
    auto start_time = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < N_test; ++i) {
      recursion(tree.back());
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "Time taken: " << duration << " milliseconds" << std::endl;
  }

  {
    auto start_time = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < N_test; ++i) {
      iteration(tree.back());
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "Time taken: " << duration << " milliseconds" << std::endl;
  }
}
