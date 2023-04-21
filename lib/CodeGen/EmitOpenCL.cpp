#include "TCC/CodeGen.h"

#include "TCC/Dialect/TBLAS/IR/TBLASOps.h"

#include "TCC/Dialect/TC/IR/TCOps.h"

#include "TCC/Utils/UnionFind.h"

#include "TCC/Dialect/TC/Utils/AxesMap.h"
#include "TCC/Dialect/TC/Utils/Utils.h"
#include "TCC/Dialect/Utils/FuncUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tc-gen-opencl"

using namespace mlir;
using namespace mlir::tc;
using namespace mlir::tc::utils;

namespace {
class EmitOpenCLGroupKernel : public codegen::CodeGenBase {
private:
  ModuleOp moduleOp;

public:
  EmitOpenCLGroupKernel(ModuleOp moduleOp, llvm::raw_ostream &out)
      : CodeGenBase(out), moduleOp(moduleOp) {}

  void emit() {
    auto funcOps = moduleOp.getOps<func::FuncOp>();
    for (auto funcOp : funcOps) {
      if (getWorkloadType(funcOp) == 2)
        emitWorkloadFunc(funcOp);
    }
    for (auto funcOp : funcOps) {
      if (getWorkloadType(funcOp) == 1)
        emitEntryFunc(funcOp);
    }
  }

private:
  int indent = 0;
  using ValueNameMapTy = DenseMap<Value, std::string>;
  ValueNameMapTy valueNameMap;
  using SizeIDValueMapTy = DenseMap<int64_t, std::string>;
  SizeIDValueMapTy sizeIDValueMap;

  void emitFunctionType(func::FuncOp funcOp) {
    std::string funcDeclare = "__kernel void ";
    out << funcDeclare;
    indent = funcDeclare.size();
    out << funcOp.getName();
    indent += funcOp.getName().size();
    out << "(";
    indent += 1;
    bool newLine = false;
    emitArgumentTypes(funcOp.front(), newLine);
    emitResultTypes(funcOp.front(), newLine);
    out << ")\n";
  }

  void emitArgumentTypes(Block &block, bool &newLine) {
    auto inputs = block.getArguments();
    for (auto [idx, input] : llvm::enumerate(inputs)) {
      if (newLine) {
        out << ",\n";
        out.indent(indent);
      }
      newLine = true;
      auto name = "arg" + std::to_string(idx);
      if (input.getType().isIndex()) {
        out << "int " << name;
        sizeIDValueMap[idx] = name;
      } else {
        out << "double* " << name;
      }
      valueNameMap[input] = name;
    }
  }

  SmallVector<std::string> returnNames;
  void emitResultTypes(Block &block, bool &newLine) {
    auto returnOp = block.getTerminator();
    for (auto [idx, input] : llvm::enumerate(returnOp->getOperands())) {
      if (newLine) {
        out << ",\n";
        out.indent(indent);
      }
      newLine = true;
      auto name = "ret" + std::to_string(idx);
      out << "double* " << name;
      valueNameMap[input] = name;
      returnNames.emplace_back(name);
    }
  }

  void emitWorkloadFunctionBody(func::FuncOp funcOp) {
    out << "{\n";
    indent = 2;
    auto &block = funcOp.front();
    auto returnOp = block.getTerminator();
    assert(returnOp->getOperands().size() == 1);
    auto result = returnOp->getOperand(0);
    DenseMap<Operation *, size_t> operationUsesSize;
    for (auto &op : block) {
      if (op.getResults().size() == 0)
        continue;
      operationUsesSize[&op] = 0;
      for (auto input : op.getOperands()) {
        if (auto op = input.getDefiningOp())
          operationUsesSize[op]++;
      }
    }
    DenseMap<Operation *, SmallVector<AxesTy>> operationLoopAxes;
    SmallVector<Operation *> worklist;
    auto addOperationToWorklist = [&](Operation *op) {
      if (operationLoopAxes[op].size() == operationUsesSize[op] &&
          llvm::all_equal(operationLoopAxes[op])) {
        operationLoopAxes[op].truncate(1);
        worklist.emplace_back(op);
      }
    };
    auto op = result.getDefiningOp();
    assert(op);
    worklist.emplace_back(op);
    auto loopSizeIDs = getTensorSizeIDs(result);
    operationLoopAxes[op].emplace_back(
        llvm::to_vector(llvm::seq<AxisTy>(0, loopSizeIDs.size())));
    while (!worklist.empty()) {
      auto op = worklist.pop_back_val();
      auto loopAxes = AxesRefTy(operationLoopAxes[op][0]);
      auto addInputWithAxesMap = [&](Value input, AxesMapRefTy axesMap) {
        if (auto inputOp = input.getDefiningOp()) {
          auto inputLoopAxes = applyAxesMap(loopAxes, axesMap);
          operationLoopAxes[inputOp].emplace_back(inputLoopAxes);
          addOperationToWorklist(inputOp);
        }
      };
      if (auto addOp = dyn_cast<AddOp>(op)) {
        auto inputs = addOp.getInputs();
        auto axesMaps = addOp.getAxesMaps();
        for (auto [input, axesMap] : llvm::zip(inputs, axesMaps))
          addInputWithAxesMap(input, axesMap);
      } else if (auto mulOp = dyn_cast<MulOp>(op)) {
        auto inputs = mulOp.getInputs();
        auto axesMaps = mulOp.getAxesMaps();
        for (auto [input, axesMap] : llvm::zip(inputs, axesMaps))
          addInputWithAxesMap(input, axesMap);
      } else if (auto ctOp = dyn_cast<ContractOp>(op)) {
        auto input = ctOp.getInput();
        if (auto inputOp = input.getDefiningOp()) {
          auto axes = ctOp.getAxes();
          auto sizeIDs = getTensorSizeIDs(input);
          auto inputLoopAxes = getContractOpInput2OutputAxisMap(ctOp);
          applyAxesMapWithEmpty(loopAxes, inputLoopAxes);
          for (auto axis : axes) {
            inputLoopAxes[axis] = loopSizeIDs.size();
            loopSizeIDs.emplace_back(sizeIDs[axis]);
          }
          operationLoopAxes[inputOp].emplace_back(inputLoopAxes);
          addOperationToWorklist(inputOp);
        }
      } else if (auto permuteOp = dyn_cast<PermuteOp>(op)) {
        auto input = permuteOp.getInput();
        auto axesMaps = permuteOp.getAxesMap();
        addInputWithAxesMap(input, axesMaps[0]);
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "operationLoopAxes size is: "
                            << operationLoopAxes.size() << "\n";
               llvm::dbgs() << "operationUsesSize size is: "
                            << operationUsesSize.size() << "\n";);
    LLVM_DEBUG(llvm::dbgs()
                   << "Number of loops: " << loopSizeIDs.size() << "\n";
               llvm::dbgs() << "    loop ranges is [";
               llvm::interleaveComma(loopSizeIDs, llvm::dbgs());
               llvm::dbgs() << "]\n";);
    LLVM_DEBUG(llvm::dbgs() << "\n";);
    if (operationLoopAxes.size() != operationUsesSize.size())
      llvm_unreachable("TODO: split workload into more small piece.");

    out << "}\n";
  }

  void emitWorkloadFunc(func::FuncOp funcOp) {
    out << "inline\n";
    emitFunctionType(funcOp);
    emitWorkloadFunctionBody(funcOp);
    out << "\n";
  }

  void emitEntryFunc(func::FuncOp funcOp) {
    out << "// TODO: Emit entry " << funcOp.getName() << "\n";
  }
};
} // namespace

namespace mlir::tc::codegen {
void emitTBLASToOpenCL(ModuleOp op, llvm::raw_ostream &out, bool workgroup) {
  if (!workgroup)
    llvm_unreachable("TODO: emit tblas to OpenCL kernel.\n");
  EmitOpenCLGroupKernel(op, out).emit();
}
} // namespace mlir::tc::codegen
