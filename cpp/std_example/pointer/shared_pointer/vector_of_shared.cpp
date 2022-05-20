#include <algorithm>
#include <memory>
#include <vector>
#include <iostream>


struct Node {
  size_t idx;
};


using NodePtr = std::shared_ptr<Node>;

int main () {

  NodePtr target;
  std::vector<NodePtr> nodes;
  for (size_t i = 0; i < 10; ++i) {
    const auto node_new = std::make_shared<Node>(Node{i});
    nodes.push_back(node_new);
    if (i == 3) {
      target = node_new;
    }
  }

  const auto f_pred = [&target](const NodePtr node) -> bool { return node.get() == target.get(); };
  const auto it = std::find_if(nodes.begin(), nodes.end(), f_pred);
  std::cout << (*it)->idx << std::endl;
}
