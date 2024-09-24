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

  class StackMachineInstruction{
    enum class Kind { PUSH, ADD, SUB, MUL };
    Kind kind;
    std::string value;
  };

  void compile(std::vector<Operation::Ptr>& inputs){
    std::vector<Operation::Ptr> instructions;
    std::stack<Operation::Ptr> stack;
    stack.push(shared_from_this());
    while(!stack.empty()){
      auto op = stack.top();
      stack.pop();
      if(op->kind != OpKind::VALUE){
        instructions.push_back(op);
        stack.push(op->lhs);
        stack.push(op->rhs);
      }
    }

    std::vector<Operation::Ptr> xmm_registers(16, nullptr);
    for(size_t i = 0; i < inputs.size(); ++i){
      xmm_registers[i] = inputs[i];
    }

    auto smallest_available_register = [&xmm_registers]() -> int{
      for(size_t i = 0; i < xmm_registers.size(); ++i){
        if(xmm_registers[i] == nullptr){
          return i;
        }
      }
      return -1;
    };

    auto xmm_index_of_op = [&xmm_registers](Operation::Ptr op) -> int{
      for(size_t i = 0; i < xmm_registers.size(); ++i){
        if(xmm_registers[i] == op){
          return i;
        }
      }
      throw std::runtime_error("error");
    };

    auto visualize_registers = [&xmm_registers](){
      // single line. if not set, print nil
      for(size_t i = 0; i < xmm_registers.size(); ++i){
        if(xmm_registers[i] == nullptr){
          std::cout << "xmm" << i << ": nil, ";
        } else {
          std::cout << "xmm" << i << ": " << xmm_registers[i]->name << ", ";
        }
      }
      std::cout << std::endl;
    };

    // reverse order
    // actually, if stack-machine is used, no need to check if the register is used by other
    for(auto it = instructions.rbegin(); it != instructions.rend(); ++it){
      auto instr = *it;
      if(instr->kind == OpKind::VALUE){
        continue;
      }
      auto lhs_index = xmm_index_of_op(instr->lhs);
      auto rhs_index = xmm_index_of_op(instr->rhs);

      bool is_required_by_other = false;
      auto result_index = lhs_index;
      if(instr->kind == OpKind::ADD){
        std::cout << "vaddss xmm" << result_index << ", xmm" << lhs_index << ", xmm" << rhs_index << std::endl;
      } else if(instr->kind == OpKind::SUB){
        std::cout << "vsubss xmm" << result_index << ", xmm" << lhs_index << ", xmm" << rhs_index << std::endl;
      } else if(instr->kind == OpKind::MUL){
        std::cout << "vmulss xmm" << result_index << ", xmm" << lhs_index << ", xmm" << rhs_index << std::endl;
      }
      xmm_registers[result_index] = instr;
      xmm_registers[rhs_index] = nullptr;
      // visualize_registers();
    }
    size_t idx_ret_op = xmm_index_of_op(shared_from_this());
    if(idx_ret_op != 0){
      std::cout << "movss xmm0, xmm" << idx_ret_op << std::endl;
    }
    std::cout << "ret" << std::endl;
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
  auto f = Operation::make_value("f");
  auto g = Operation::make_value("g");
  auto h = (a + b) * (c - d) + (e * f + g);
  // auto g = (a + b) - (c + d) + f;
  auto inputs = std::vector<Operation::Ptr>{a, b, c, d, e, f, g};
  h->compile(inputs);
}
