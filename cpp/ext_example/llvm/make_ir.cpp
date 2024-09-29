#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>

// DLLVM_ENABLE_DUMP=ON


int main(){
  llvm::LLVMContext context;
  llvm::Module llvm_module("top", context);
  llvm::IRBuilder<> builder(context);

  llvm::FunctionType* return_type = llvm::FunctionType::get(builder.getInt32Ty(), false);
  llvm::Function* fn = llvm::Function::Create(return_type, llvm::Function::ExternalLinkage, "main", llvm_module);
  llvm::BasicBlock* bb = llvm::BasicBlock::Create(context, "entry", fn);
  builder.SetInsertPoint(bb);

  llvm::Value* x = builder.getInt32(1);
  llvm::Value* y = builder.getInt32(2);
  llvm::Value* result = builder.CreateAdd(x, y, "result");
  builder.CreateRet(result);

  llvm_module.print(llvm::errs(), nullptr);

}
