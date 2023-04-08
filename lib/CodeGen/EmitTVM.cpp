#include "TCC/CodeGen.h"
#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Utils/Utils.h"
#include "TCC/Dialect/Utils/FuncUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;
using namespace mlir::tc;
using namespace mlir::tc::codegen;

namespace {
class EmitTVMImpl : public CodeGenBase {
  func::FuncOp funcOp;
  Block &block;
  int numInputSizes;

public:
  EmitTVMImpl(func::FuncOp funcOp, llvm::raw_ostream &out)
      : CodeGenBase(out), funcOp(funcOp), block(funcOp.front()) {
    numInputSizes = utils::getNumInputSizes(funcOp);
  }

  void emit() {
    emitImports();
    out << "\n\n";
    emitFunction();
    out << "\n\n";
    emitCreateTask();
    out << "\n\n";
    emitTesting();
  }

private:
  int indent = 0;
  using ValueNameMapTy = DenseMap<Value, std::string>;
  ValueNameMapTy valueNameMap;

  void emitImports() {
    out << "#!python\n";
    out << "from ..python.common import *\n";
    out << "from tvm import te, topi, auto_scheduler\n";
  }

  void emitCreateTask() {
    out << "def getTask(target):\n";
    indent += 4;
    auto [lbs, ubs] = utils::getSizesRanges(funcOp);
    for (auto idx : llvm::seq<int>(0, numInputSizes)) {
      out.indent(indent) << "s" << idx << " = ";
      if (ubs[idx] == 0)
        out << 256;
      else if (lbs[idx] == ubs[idx])
        out << lbs[idx];
      else {
        auto temp = lbs[idx] + ubs[idx];
        if (temp > 256)
          temp = (temp + 15LL) & (-16LL);
        out << temp / 2;
      }
      out << "\n";
    }
    out.indent(indent) << "log_file = '" << funcOp.getName()
                       << "' + '_' + target + '.json'\n";
    out.indent(indent) << "task = auto_scheduler.SearchTask(func=";
    out << funcOp.getName() << ", args=(";
    for (auto idx : llvm::seq<int>(0, numInputSizes))
      out << "s" << idx << ", ";
    out << "), target=target)\n";
    out.indent(indent) << "return task, log_file\n";
    indent -= 4;
  }

  void emitTesting() {
    out << "for target in targets:\n";
    indent += 4;
    out.indent(indent) << "task, log_file = getTask(target)\n";
    out.indent(indent) << "print_compute_dag(task)\n";
    out.indent(indent) << "tune_task(task, log_file)\n";
    out.indent(indent) << "time_task(task, log_file, target)\n";
    indent -= 4;
  }

  void emitFunction() {
    out << "@auto_scheduler.register_workload\n";
    out << "def " << funcOp.getName() << "(";
    emitInputSizes();
    out << "):\n";
    indent += 4;
    emitInputTensors();
    emitFuncionBody();
    indent -= 4;
  }

  void emitInputSizes() {
    for (auto idx : llvm::seq<int>(0, numInputSizes)) {
      if (idx != 0) {
        out << ", ";
      }
      out << "s" << idx;
    }
  }

  SmallVector<std::string> inputTensorNames;
  void emitInputTensors() {
    for (auto argValue : funcOp.getArguments()) {
      if (argValue.getType().isa<RankedTensorType>()) {
        out.indent(indent) << "# ";
        out << utils::getFuncOpArgName(funcOp, argValue.getArgNumber());
        out << "\n";
        std::string name = "t" + std::to_string(argValue.getArgNumber());
        out.indent(indent) << name << " = te.placeholder(";
        emitTensorSizes(argValue);
        emitTEDtypeArgument();
        emitTENameArgument(name);
        valueNameMap[argValue] = name;
        inputTensorNames.emplace_back(name);
      }
    }
  }

  void emitTensorSizes(Value value) {
    auto [shape, sizeIDs] = getTensorSizeInfos(value);
    out << "(";
    for (auto [idx, size] : llvm::enumerate(shape)) {
      if (idx != 0) {
        out << ", ";
      }
      emitSizeID(size, sizeIDs[idx]);
    }
    out << ",)";
  }

  void emitFuncionBody() {
    for (auto &op : block) {
      if (auto addOp = dyn_cast_or_null<AddOp>(&op)) {
        emitTCAddOp(addOp);
      } else if (auto mulOp = dyn_cast_or_null<MulOp>(&op)) {
        emitTCMulOp(mulOp);
      } else if (auto ctOp = dyn_cast_or_null<ContractOp>(&op)) {
        emitTCContractOp(ctOp);
      } else if (auto pmOp = dyn_cast_or_null<PermuteOp>(&op)) {
        emitTCPermuteOp(pmOp);
      } else if (auto cstOp = dyn_cast_or_null<ConstantOp>(&op)) {
        emitTCConstantOp(cstOp);
      } else if (auto returnOp = dyn_cast_or_null<ReturnOp>(&op)) {
        emitFuncReturnOp(returnOp);
      } else {
        out.indent(indent) << "# TODO: " << op.getName() << "\n";
      }
    }
  }

  void emitSizeID(int64_t size, int64_t sizeID) {
    if (size >= 0) {
      out << size;
    } else {
      out << "s" << sizeID;
    }
  }

  int varIdx = 0;
  void emitTEComputeForValue(Value value) {
    std::string name = "var" + std::to_string(varIdx++);
    out.indent(indent) << name << " = te.compute(";
    emitTensorSizes(value);
    out << ", lambda *ls: ";
    valueNameMap[value] = name;
  }
  void emitTOPISumForValue(Value value) {
    std::string name = "var" + std::to_string(varIdx++);
    out.indent(indent) << name << " = topi.sum(";
    valueNameMap[value] = name;
  }

  void emitTEDtypeArgument() { out << ", dtype=\"float64\""; }
  void emitTENameArgument(Value value) {
    emitTENameArgument(valueNameMap[value]);
  }
  void emitTENameArgument(std::string name) {
    out << ", name=\"" << name << "\")\n";
  }

  void emitTCMulOp(MulOp op) {
    auto inputs = op.getInputs();
    auto result = op.getResult();
    auto axesMapsRef = op.getAxesMaps();
    emitTEComputeForValue(result);
    for (auto [idx, input] : llvm::enumerate(inputs)) {
      if (idx != 0) {
        out << " * ";
      }
      out << valueNameMap[input];
      for (auto axis : axesMapsRef[idx]) {
        out << "[ls[" << axis << "]]";
      }
    }
    emitTENameArgument(result);
  }

  void emitTCAddOp(AddOp op) {
    auto inputs = op.getInputs();
    auto result = op.getResult();
    auto axesMapsRef = op.getAxesMaps();
    emitTEComputeForValue(result);
    for (auto [idx, input] : llvm::enumerate(inputs)) {
      if (idx != 0) {
        out << " + ";
      }
      out << valueNameMap[input];
      for (auto axis : axesMapsRef[idx]) {
        out << "[ls[" << axis << "]]";
      }
    }
    emitTENameArgument(result);
  }

  int reduceAxisIdx = 0;
  std::string emitTEReductAxis(int64_t size, int64_t sizeIDs) {
    std::string name = "a" + std::to_string(reduceAxisIdx++);
    out.indent(indent) << name << " = te.reduce_axis((0, ";
    emitSizeID(size, sizeIDs);
    out << ")";
    emitTENameArgument(name);
    return name;
  }

  void emitTCContractOp(ContractOp op) {
    auto input = op.getInput();
    auto result = op.getResult();
    auto axes = op.getAxes();
    emitTOPISumForValue(result);
    out << valueNameMap[input];
    out << ", axis=[";
    llvm::interleaveComma(axes, out);
    out << "])\n";
  }

  void emitTCPermuteOp(PermuteOp op) {
    auto input = op.getInput();
    auto result = op.getResult();
    auto axesMap = op.getAxesMapArray();
    emitTEComputeForValue(result);
    out << valueNameMap[input];
    for (auto axis : axesMap) {
      out << "[ls[" << axis << "]]";
    }
    emitTENameArgument(result);
  }

  int cstIdx = 0;
  void emitTCConstantOp(ConstantOp op) {
    std::string name = "cst" + std::to_string(cstIdx++);
    SmallVector<char> value;
    op.getValue().toString(value);
    out.indent(indent) << name << " = " << value << "\n";
    valueNameMap[op.getResult()] = name;
  }

  void emitFuncReturnOp(ReturnOp op) {
    out.indent(indent) << "return [";
    llvm::interleaveComma(inputTensorNames, out);
    auto inputs = op.getOperands();
    out << ", ";
    llvm::interleaveComma(inputs, out,
                          [&](Value value) { out << valueNameMap[value]; });
    out << "]\n";
  }
};
} // namespace

namespace mlir::tc::codegen {
void emitTCToTVM(ModuleOp op, llvm::raw_ostream &out) {
  SymbolTable symTable(op);
  auto funcOps = op.getOps<func::FuncOp>();
  for (auto funcOp : funcOps) {
    EmitTVMImpl(funcOp, out).emit();
  }
}
} // namespace mlir::tc::codegen
