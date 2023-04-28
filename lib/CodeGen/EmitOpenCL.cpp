#include "TCC/CodeGen.h"

#include "TCC/Dialect/TBLAS/IR/TBLASOps.h"

#include "TCC/Dialect/TC/IR/TCOps.h"

#include "TCC/Utils/UnionFind.h"

#include "TCC/Dialect/TC/Utils/AxesMap.h"
#include "TCC/Dialect/TC/Utils/Utils.h"
#include "TCC/Dialect/Utils/FuncUtils.h"
#include "TCC/Utils/SetRelation.h"

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
  ArrayRef<int64_t> lbs, ubs;

public:
  EmitOpenCLGroupKernel(ModuleOp moduleOp, llvm::raw_ostream &out)
      : CodeGenBase(out), moduleOp(moduleOp) {}

  void emit() {
    auto funcOps = moduleOp.getOps<func::FuncOp>();
    func::FuncOp entryFuncOp;
    for (auto funcOp : funcOps) {
      if (getWorkloadType(funcOp) == 1)
        entryFuncOp = funcOp;
    }
    auto [lbs, ubs] = getSizesRanges(entryFuncOp);
    this->lbs = lbs;
    this->ubs = ubs;
    for (auto funcOp : funcOps) {
      if (getWorkloadType(funcOp) == 2)
        emitWorkloadFunc(funcOp);
    }
    emitEntryFunc(entryFuncOp);
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
    auto numInputSizes = getNumInputSizes(funcOp);
    for (auto sizeID : llvm::seq<int64_t>(0, numInputSizes)) {
      if (!sizeIDValueMap.count(sizeID))
        sizeIDValueMap[sizeID] = std::to_string(ubs[sizeID]);
    }
    emitResultTypes(funcOp.front(), newLine);
    out << ")\n";
  }

  void emitArgumentTypes(Block &block, bool &newLine) {
    sizeIDValueMap.clear();
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
    auto rank = loopSizeIDs.size();
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
    if (operationLoopAxes.size() != operationUsesSize.size())
      llvm_unreachable("TODO: split workload into more small piece.");
    // Outer loop
    AxesTy outerLoopAxes;
    for (auto loopAxis : llvm::seq<AxisTy>(0, rank)) {
      bool isOuter = true;
      for (auto &[op, _loopAxes] : operationLoopAxes) {
        auto &loopAxes = _loopAxes[0];
        if (!loopAxes.empty() &&
            llvm::find(loopAxes, loopAxis) == loopAxes.end()) {
          isOuter = false;
          break;
        }
      }
      if (isOuter)
        outerLoopAxes.emplace_back(loopAxis);
    }
    llvm::sort(outerLoopAxes);
    LLVM_DEBUG(llvm::dbgs()
                   << "Number of outer loops: " << outerLoopAxes.size() << "\n";
               if (outerLoopAxes.size()) {
                 llvm::dbgs() << "    outer loops are ";
                 llvm::interleaveComma(outerLoopAxes, llvm::dbgs());
                 llvm::dbgs() << "\n";
               });
    if (outerLoopAxes.empty())
      llvm_unreachable("TODO: split workload into more small piece.");
    using OperationsTy = SetVector<Operation *>;
    SmallVector<OperationsTy> loopAxisOperations(loopSizeIDs.size());
    for (auto &[op, _loopAxes] : operationLoopAxes) {
      auto &loopAxes = _loopAxes[0];
      for (auto loopAxis : loopAxes)
        loopAxisOperations[loopAxis].insert(op);
    }
    SmallVector<SmallVector<int64_t>> loopOuterLoops(loopSizeIDs.size());
    SmallVector<SmallVector<int64_t>> loopInnerLoops(loopSizeIDs.size());
    SmallVector<SmallVector<int64_t>> loopExchangableLoops(loopSizeIDs.size());
    for (auto [loopAxis1, operationSet1] :
         llvm::enumerate(loopAxisOperations)) {
      for (auto loopAxis2 : llvm::seq<AxisTy>(loopAxis1, loopSizeIDs.size())) {
        auto &operationSet2 = loopAxisOperations[loopAxis2];
        if (isIntersect(operationSet1, operationSet2)) {
          auto subset1 = isSubsetOf(operationSet1, operationSet2);
          auto subset2 = isSubsetOf(operationSet2, operationSet1);
          if (subset1 && subset2) {
            loopExchangableLoops[loopAxis1].emplace_back(loopAxis2);
            loopExchangableLoops[loopAxis2].emplace_back(loopAxis1);
          } else if (subset1) {
            loopOuterLoops[loopAxis1].emplace_back(loopAxis2);
            loopInnerLoops[loopAxis2].emplace_back(loopAxis1);
          } else if (subset2) {
            loopOuterLoops[loopAxis2].emplace_back(loopAxis1);
            loopInnerLoops[loopAxis1].emplace_back(loopAxis2);
          } else if (loopAxis1 < rank && (size_t)loopAxis2 < rank) {
            LLVM_DEBUG(
                llvm::dbgs() << "loopAxis1 = " << loopAxis1
                             << ", loopAxis2 = " << loopAxis2 << "\n";
                llvm::dbgs() << "Set commons:\n"; for (auto *op
                                                       : operationSet1) {
                  if (operationSet2.count(op)) {
                    op->dump();
                    llvm::dbgs() << "    loop axis: ";
                    llvm::interleaveComma(operationLoopAxes[op][0],
                                          llvm::dbgs());
                    llvm::dbgs() << "\n";
                  }
                } llvm::dbgs() << "Set 1 extra:\n";
                for (auto *op
                     : operationSet1) {
                  if (!operationSet2.count(op)) {
                    op->dump();
                    llvm::dbgs() << "    loop axis: ";
                    llvm::interleaveComma(operationLoopAxes[op][0],
                                          llvm::dbgs());
                    llvm::dbgs() << "\n";
                  }
                } llvm::dbgs()
                << "Set 2 extra:\n";
                for (auto *op
                     : operationSet2) {
                  if (!operationSet1.count(op)) {
                    op->dump();
                    llvm::dbgs() << "    loop axis: ";
                    llvm::interleaveComma(operationLoopAxes[op][0],
                                          llvm::dbgs());
                    llvm::dbgs() << "\n";
                  }
                });
            llvm_unreachable("TODO: split workload into more small piece.");
          }
        }
      }
    }
    LLVM_DEBUG(for (auto [loopAxis, outerLoops]
                    : llvm::enumerate(loopOuterLoops)) {
      llvm::dbgs() << "Loop axis: " << loopAxis << ", outer loops are [";
      llvm::interleaveComma(outerLoops, llvm::dbgs());
      llvm::dbgs() << "]\n";
    });
    for (auto [idx, sizeID] : llvm::enumerate(loopSizeIDs)) {
      out.indent(indent) << "// i" << idx << " < " << sizeIDValueMap[sizeID] << "\n";
    }
    AxisTy loopAxis = outerLoopAxes[0];
    std::string name = "i" + std::to_string(loopAxis);
    out.indent(indent) << "int i" << loopAxis << " = get_local_id(0);\n";
    auto sizeID = loopSizeIDs[loopAxis];
    out.indent(indent) << "if (" << name << " >= " << sizeIDValueMap[sizeID]
                       << ") return;\n";
    for (auto innerLoop : loopExchangableLoops[loopAxis]) {
      if (innerLoop == loopAxis)
        continue;
      auto sizeID = loopSizeIDs[innerLoop];
      std::string name = "i" + std::to_string(innerLoop);
      out.indent(indent) << "for (int " << name << " = 0; ";
      out << name << " < " << sizeIDValueMap[sizeID] << "; ";
      out << name << "++) {\n";
      indent += 2;
    }
    int varId = 0;
    for (auto &op : block) {
      if (auto constOp = llvm::dyn_cast<ConstantOp>(&op)) {
        SmallVector<char> value;
        constOp.getValue().toString(value);
        valueNameMap[constOp.getResult()] =
            std::string(value.begin(), value.end());
        continue;
      }
      if (auto returnOp = llvm::dyn_cast<ReturnOp>(&op))
        continue;
      auto emitNewVar = [&]() {
        std::string name = "var" + std::to_string(varId++);
        out.indent(indent) << name << "(";
        llvm::interleaveComma(operationLoopAxes[&op][0], out,
                              [&](AxisTy axis) { out << "i" << axis; });
        out << ") = ";
        valueNameMap[op.getResult(0)] = name;
      };
      if (auto addOp = llvm::dyn_cast<AddOp>(&op)) {
        emitNewVar();
        auto inputs = addOp.getInputs();
        auto axesMaps = addOp.getAxesMaps();
        for (auto [idx, input] : llvm::enumerate(inputs)) {
          if (idx != 0)
            out << " + ";
          out << valueNameMap[input];
          out << "(";
          auto axesMap = axesMaps[idx];
          applyAxesMap(operationLoopAxes[&op][0], axesMap);
          llvm::interleaveComma(axesMap, out,
                                [&](AxisTy axis) { out << "i" << axis; });
          out << ")";
        }
        out << ";\n";
      } else if (auto mulOp = llvm::dyn_cast<MulOp>(&op)) {
        emitNewVar();
        auto inputs = mulOp.getInputs();
        auto axesMaps = mulOp.getAxesMaps();
        for (auto [idx, input] : llvm::enumerate(inputs)) {
          if (idx != 0)
            out << " * ";
          out << valueNameMap[input];
          out << "(";
          auto axesMap = axesMaps[idx];
          applyAxesMap(operationLoopAxes[&op][0], axesMap);
          llvm::interleaveComma(axesMap, out,
                                [&](AxisTy axis) { out << "i" << axis; });
          out << ")";
        }
        out << ";\n";
      } else if (auto ctOp = llvm::dyn_cast<ContractOp>(&op)) {
        ctOp.dump();
        std::string name = "var" + std::to_string(varId++);
        out.indent(indent) << name << "(";
        valueNameMap[op.getResult(0)] = name;
        auto ctOpMap = getContractOpInput2OutputAxisMap(ctOp);
        AxesTy resultAxes;
        for (auto axis : ctOpMap) {
          if (axis >= 0)
            resultAxes.emplace_back(operationLoopAxes[&op][0][axis]);
        }
        llvm::errs() << resultAxes.size() << "\n";
        llvm::interleaveComma(resultAxes, out,
                              [&](AxisTy axis) { out << "i" << axis; });        out << ") ";
        out << "+= ";
        auto input = ctOp.getInput();
        out << valueNameMap[input];
        out << "(";
        llvm::errs() << operationLoopAxes[&op][0].size() << "\n";
        llvm::interleaveComma(operationLoopAxes[&op][0], out,
                              [&](AxisTy axis) { out << "i" << axis; });
        out << ")";
        out << ";\n";
      } else if (auto permuteOp = dyn_cast<PermuteOp>(&op)) {
        emitNewVar();
        auto input = permuteOp.getInput();
        out << valueNameMap[input];
        out << "(";
        auto axesMap = permuteOp.getAxesMapArray();
        applyAxesMap(operationLoopAxes[&op][0], axesMap);
        llvm::interleaveComma(axesMap, out,
                              [&](AxisTy axis) { out << "i" << axis; });
        out << ")";
        out << ";\n";
      }
    }
    // TODO: init
    // SmallVector<AxisTy> nestLoops;
    // SetVector<Operation *> worklist;
    // while (!worklist.empty()) {
    //   auto &subOperations = loopAxisOperations[loopAxis];
    //   // Operation *toEmit = nullptr;
    //   // for (auto op : worklist) {
    //   //   bool canEmit = true;
    //   //   for (auto input : op->getOperands()) {
    //   //     if (!valueNameMap.count(input)) {
    //   //       canEmit = false;
    //   //       break;
    //   //     }
    //   //   }
    //   //   if (!canEmit)
    //   //     continue;
    //   //   auto opLoops = operationLoopAxes[op][0];

    //   }
    // }
    for (auto innerLoop : loopExchangableLoops[loopAxis]) {
      if (innerLoop == loopAxis)
        continue;
      indent -= 2;
      out.indent(indent) << "}\n";
    }
    LLVM_DEBUG(llvm::dbgs() << "\n";);
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
    emitFunctionType(funcOp);
    out << "{\n";
    out << "}\n";
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
