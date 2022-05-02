#include <boost/optional/optional.hpp>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <stack>
#include <unordered_set>
#include <unordered_map>
#include <boost/optional.hpp>

template <typename ElementT>
bool isInside(ElementT elem, std::unordered_set<ElementT> elemset) {
  return elemset.find(elem) != elemset.end();
}

template<typename T>
using VecVec = std::vector< std::vector<T> >;

void print_set(std::unordered_set<size_t> elemset) {
  for (size_t e : elemset) {
    std::cout << e << ", ";
  }
  std::cout << std::endl;
}


template <typename ElementT>
class ParkingGraphBase
{
public:
  bool hasLoop(ElementT element) const{
    std::unordered_set<size_t> visit_set;
    std::stack<ElementT> s;
    s.push(element);
    while (!s.empty()) {
      const auto elem_here = s.top();
      s.pop();
      const bool is_visisted = isInside(get_id(elem_here), visit_set);
      if (is_visisted) return true;

      visit_set.insert(elem_here.id);
      const auto elmes_following = get_followings(elem_here);
      for (const auto elem : elmes_following) {
        s.push(elem);
      }
    }
    return false;
  }

  std::vector<ElementT> computeEntireCircularPathWithLoop(ElementT element) const {
    std::unordered_set<size_t> outside_of_loop_set;
    std::unordered_map<size_t, size_t> visit_counts;
    std::vector<ElementT> circular_path;

    // lambda
    const auto isOutOfLoop = [&](const ElementT & elem) -> bool {
      if (isInside(get_id(elem), outside_of_loop_set)) return true;
      if (hasLoop(elem)) return false;

      for (const auto & elem_descendants : get_reachables(elem)) {
        outside_of_loop_set.insert(get_id(elem_descendants));
      }
      return true;
    };

    // lambda
    const auto getVisitCount = [&](const ElementT & elem) -> size_t {
      const auto id = get_id(elem);
      const bool never_visit = visit_counts.find(id) == visit_counts.end();
      if (never_visit) visit_counts[id] = 0;
      return visit_counts[id];
    };

    // lambda
    const auto incrementVisitCount = [&](const ElementT & elem) -> void {
      const auto id = get_id(elem);
      const bool never_visit = visit_counts.find(id) == visit_counts.end();
      if (never_visit) visit_counts[id] = 0;
      visit_counts[id] += 1;
    };

    // lambda
    const auto markedAllNode = [&]() -> bool {
      size_t c = 0;
      for (const auto p : visit_counts) {
        if (p.second > 0) c++;
      }
      const auto n_total_mark = c + outside_of_loop_set.size();
      if (n_total_mark > n_element()) throw std::logic_error("strange");
      return (n_total_mark == n_element());
    };

    // main
    ElementT element_here = element;
    while (!markedAllNode()) {

      incrementVisitCount(element_here);
      circular_path.push_back(element_here);

      const auto elems_child = get_followings(element_here);

      ElementT element_next;
      const bool is_forking = elems_child.size() > 1;
      if (!is_forking) {
        const auto elem_next_cand = elems_child.front();
        if (isOutOfLoop(elem_next_cand)) throw std::logic_error("strange");
        element_next = elem_next_cand;
      } else {
        boost::optional<size_t> best_idx = boost::none;
        size_t min_visit_count = std::numeric_limits<size_t>::max();
        for (size_t idx = 0; idx < elems_child.size(); ++idx) {
          const auto & elem_next = elems_child.at(idx);

          if (isOutOfLoop(elem_next)) continue;

          const auto visit_count = getVisitCount(elem_next);
          if (visit_count < min_visit_count) {
            best_idx = idx;
            min_visit_count = visit_count;
          }
        }
        element_next = elems_child[best_idx.get()];
      }
      element_here = element_next;
    }
    return circular_path;
  };

  VecVec<ElementT> splitPathContainingLoop(std::vector<ElementT> elem_path) const{

    std::vector<std::vector<ElementT>> partial_path_seq;

    std::vector<ElementT> partial_path;
    std::unordered_set<size_t> partial_visit_set;

    for (const auto & elem : elem_path) {
      const auto is_tie = isInside(get_id(elem), partial_visit_set);

      if (is_tie) { // split the loop!
        partial_path_seq.push_back(partial_path);

        // Initialize partial_path and partial_visit_set
        partial_path = std::vector<ElementT>{partial_path.back()}; // Later will be reversed
        //partial_visit_set = std::unordered_set<size_t>{partial_path.back()};
        partial_visit_set = {get_id(partial_path.back())};
        // add hook procedure ...
        
        std::reverse(partial_path.begin(), partial_path.end());
      }

      partial_path.push_back(elem);
      partial_visit_set.insert(get_id(elem));
    }

    partial_path_seq.push_back(partial_path);
    return partial_path_seq;
  }

private:
  virtual std::vector<ElementT> get_followings(const ElementT & element) const = 0;
  virtual std::vector<ElementT> get_reachables(const ElementT & element) const = 0;
  virtual size_t get_id(const ElementT & element) const = 0;
  virtual size_t n_element() const = 0;
};

struct Node {
  size_t id;
  std::vector<size_t> child_ids;
  void add_child(size_t id) { 
    child_ids.push_back(id);
  }
};


class SimpleGraph : public ParkingGraphBase<Node>
{

public:

  explicit SimpleGraph(const std::vector<Node> & nodes) : nodes_(nodes) {}

  size_t get_id(const Node & node) const {
    return node.id;
  }

  size_t n_element() const {
    return nodes_.size();
  }

  std::vector<Node> get_followings(const Node & node) const{
    std::vector<Node> child_nodes;
    for (size_t id : node.child_ids) { 
      child_nodes.push_back(nodes_[id]);
    }
    return child_nodes;
  };

  std::vector<Node> get_reachables(const Node & node) const { 
    std::unordered_set<size_t> visited_set;
    std::vector<Node> reachables;
    std::stack<Node> s;
    s.push(node);
    while (!s.empty()) {
      const auto node_here = s.top();
      reachables.push_back(node_here);
      visited_set.insert(node_here.id);
      s.pop();
      for (const auto& node_child : get_followings(node_here)) {
        if (isInside(node_child.id, visited_set)) {
          continue;
        }
        s.push(node_child);
      }
    }
    return reachables;
  }

  std::vector<Node> nodes_;

};

int main(){
  std::vector<Node> nodes;
  for (size_t i = 0; i < 14; i++) {
    nodes.push_back(Node{i});
  }
  nodes[0].add_child(1);
  nodes[1].add_child(2);
  nodes[2].add_child(3);
  nodes[3].add_child(4);
  nodes[3].add_child(8);
  nodes[4].add_child(5);
  nodes[5].add_child(6);
  nodes[6].add_child(7);
  nodes[7].add_child(9);
  nodes[8].add_child(7);
  nodes[9].add_child(10);
  nodes[10].add_child(11);
  nodes[11].add_child(2);
  nodes[11].add_child(12);
  nodes[12].add_child(13);
  const auto graph = SimpleGraph(nodes);

  // test for "test class"
  for(size_t i = 0; i < 14 ; ++i) {
    if (i == 3) {
      assert(graph.get_followings(nodes[i]).size() == 2);
    } else if (i == 11) {
      assert(graph.get_followings(nodes[i]).size() == 2);
    } else if (i == 13) {
      assert(graph.get_followings(nodes[i]).size() == 0);
    } else {
      assert(graph.get_followings(nodes[i]).size() == 1);
    }
  }
  for (size_t i = 0; i < 3; ++i) {
    assert(graph.get_reachables(nodes[i]).size() == 14 - i);
  }
  for (size_t i = 3; i < 12; ++i) {
    assert(graph.get_reachables(nodes[i]).size() == 12);
  }
  for (size_t i = 12; i < 14; ++i) {
    assert(graph.get_reachables(nodes[i]).size() == 14 - i);
  }


  // main test
  for (size_t i = 0; i < 12; ++i) {
    assert(graph.hasLoop(nodes[i]));
  }
  for (size_t i = 12; i < 14; ++i) {
    assert(!graph.hasLoop(nodes[i]));
  }

  const auto entire_path = graph.computeEntireCircularPathWithLoop(nodes[0]);
  {
    std::vector<size_t> idseq_expected{0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 2, 3, 8};
    assert(entire_path.size() == idseq_expected.size());
    for (size_t i = 0; i < entire_path.size(); ++i) {
      assert(entire_path.at(i).id == idseq_expected.at(i));
    }
  }

  const auto partial_path_seq = graph.splitPathContainingLoop(entire_path);
  {
    std::vector<size_t> idseq_expected1{0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11};
    std::vector<size_t> idseq_expected2{11, 2, 3, 8};

    assert(partial_path_seq.at(0).size() == idseq_expected1.size());
    assert(partial_path_seq.at(1).size() == idseq_expected2.size());
  }
}
