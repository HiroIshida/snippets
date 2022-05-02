#include <boost/optional/optional.hpp>
#include <vector>
#include <boost/optional.hpp>

template <typename ElementT>
class ParkingGraphBase
{
  virtual std::vector<ElementT> get_followings(ElementT element) = 0;

  //bool hasLoopFromHere()
};

struct Node {
  size_t id;
  std::vector<size_t> child_ids;
  void add_child(size_t id) { 
    child_ids.push_back(id);
  }
};


class SimpleGraph : ParkingGraphBase<Node>
{

public:

  SimpleGraph(const std::vector<Node> & nodes) : nodes_(nodes) {}
  std::vector<Node> nodes_;

};

int main(){
  std::vector<Node> nodes;
  for (size_t i = 0; i < 9; i++) {
    nodes.push_back(Node{i});
  }
  nodes[0].add_child(1);
  nodes[1].add_child(2);
  nodes[2].add_child(3);
  nodes[2].add_child(5);
  nodes[3].add_child(4);
  nodes[4].add_child(5);
  nodes[5].add_child(6);
  nodes[6].add_child(7);
  nodes[7].add_child(1);
  nodes[7].add_child(8);
}
