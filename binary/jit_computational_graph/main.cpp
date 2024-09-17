#include <memory>
#include <optional>
#include <stack>
#include <string>
#include <fstream>
#include <iostream>
#include <string>
#include <random>
#include <algorithm>
#include <vector>

std::string generate_random_string(size_t length) {
  // c++ does not have a built-in random string generator?
  const std::string charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<> distribution(0, charset.size() - 1);
  std::string result;
  result.reserve(length);
  for (size_t i = 0; i < length; ++i) {
    result += charset[distribution(generator)];
  }
  return result;
}


struct Operation : std::enable_shared_from_this<Operation> {
  using Ptr = std::shared_ptr<Operation>;
  using WeakPtr = std::weak_ptr<Operation>;
  enum class OpKind { NIL, ADD, SUB, MUL, VALUE};
  OpKind kind;
  Operation::Ptr lhs;
  Operation::Ptr rhs;
  std::string name;
  std::vector<WeakPtr> requireds;

  Operation (std::string& name) : kind(OpKind::NIL), name(name) {}

  Operation () : kind(OpKind::NIL) {
    name = generate_random_string(6);
  }
  Operation (Operation::Ptr lhs, Operation::Ptr rhs, OpKind kind) : lhs(lhs), rhs(rhs), kind(kind) {
    name = generate_random_string(6);
  }

  static Operation::Ptr create(Operation::Ptr lhs, Operation::Ptr rhs, OpKind kind){
    auto op = std::make_shared<Operation>(lhs, rhs, kind);
    lhs->requireds.push_back(op);
    rhs->requireds.push_back(op);
    return op;
  }

  static Operation::Ptr make_value(std::string&& name){
    Operation::Ptr value = std::make_shared<Operation>(name);
    value->kind = OpKind::VALUE;
    return value;
  }

  static Operation::Ptr make_value(){
    Operation::Ptr value = std::make_shared<Operation>();
    value->kind = OpKind::VALUE;
    return value;
  }

  std::vector<Operation::Ptr> get_leafs(){
    std::vector<Operation::Ptr> leafs;
    auto is_added = [&leafs](Operation::Ptr op){
      return std::find(leafs.begin(), leafs.end(), op) != leafs.end();
    };
    std::stack <Operation::Ptr> stack;
    stack.push(shared_from_this());
    while(!stack.empty()){
      auto op = stack.top();
      stack.pop();
      if(op->kind == OpKind::VALUE){
        leafs.push_back(op);
      } else {
        if(!is_added(op->lhs)){
          stack.push(op->lhs);
        }
        if(!is_added(op->rhs)){
          stack.push(op->rhs);
        }
      }
    }
    return leafs;
  }

  void compile() {
    std::vector<std::string> instr;
    std::stack<Operation::Ptr> stack;
    stack.push(shared_from_this());
    while(!stack.empty()){
      auto op = stack.top();
      stack.pop();
      if(op->kind == OpKind::VALUE){
        instr.push_back("push " + op->name);
      } else {
        if (op->kind == OpKind::ADD){
          instr.push_back("add");
        } else if (op->kind == OpKind::SUB){
          instr.push_back("sub");
        } else if (op->kind == OpKind::MUL){
          instr.push_back("mul");
        }
        stack.push(op->lhs);
        stack.push(op->rhs);
      }
    }
    for(auto it = instr.rbegin(); it != instr.rend(); ++it){
      std::cout << *it << std::endl;
    }
  }
};

Operation::Ptr operator+(Operation::Ptr lhs, Operation::Ptr rhs){ return Operation::create(lhs, rhs, Operation::OpKind::ADD); }
Operation::Ptr operator-(Operation::Ptr lhs, Operation::Ptr rhs){ return Operation::create(lhs, rhs, Operation::OpKind::SUB); }
Operation::Ptr operator*(Operation::Ptr lhs, Operation::Ptr rhs){ return Operation::create(lhs, rhs, Operation::OpKind::MUL); }

int main(){
  auto a = Operation::make_value("a");
  auto b = Operation::make_value("b");
  auto c = Operation::make_value("c");
  auto d = Operation::make_value("d");
  auto e = Operation::make_value("e");
  auto f = (a + b) * (c - d) + e;
  f->compile();
}
