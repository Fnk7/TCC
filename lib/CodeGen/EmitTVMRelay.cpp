#include "TCC/CodeGen.h"
#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Utils/AxesMap.h"
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
    emitTesting();
  }

private:
  int indent = 0;
  using ValueNameMapTy = DenseMap<Value, std::string>;
  ValueNameMapTy valueNameMap;

  void emitPythonString(StringRef name) { out << "\"" << name << "\""; }

  void emitImports() {
    out << "#!python\n";
    out << "import os\n";
    out << "import sys\n";
    out << "root_dir = os.path.abspath('..')\n";
    out << "sys.path.insert(0, root_dir)\n";
    out << "from test_tvm_relay import test_tvm_relay\n";
    out << "import tvm\n";
    out << "from tvm import relay\n";
  }

  void emitTesting() {
    auto [lbs, ubs] = utils::getSizesRanges(funcOp);
    for (auto idx : llvm::seq<int>(0, numInputSizes)) {
      out << "s" << idx << " = ";
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
    out << "test_tvm_relay(get_calculation, ";
    llvm::interleaveComma(llvm::seq<int>(0, numInputSizes), out,
                          [&](int idx) { out << "s" << idx; });
    out << ")\n";
  }

  void emitFunction() {
    out << "def get_calculation(";
    emitInputSizes();
    out << "):\n";
    indent += 4;
    out.indent(indent) << "dtype = \"float64\"\n";
    out.indent(indent) << "shapes = {}\n";
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
        auto argName = utils::getFuncOpArgName(funcOp, argValue.getArgNumber());
        // shape = (s0, s1, s2,)
        out.indent(indent) << "shape = ";
        emitTensorSizes(argValue);
        out << "\n";
        // shapes["argName"] = shape
        out.indent(indent) << "shapes[";
        emitPythonString(argName);
        out << "] = shape\n";
        // t0 = t3 = relay.var("argName", shape=shape, dtype=dtype)
        std::string name = "t" + std::to_string(argValue.getArgNumber());
        out.indent(indent) << name << " = relay.var(";
        emitPythonString(argName);
        out << ", shape=shape, dtype=dtype)\n";
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

  void emitInverseAxesMap(AxesMapRefTy axesMap, int64_t rank) {
    out << "(";
    auto inverseMap = utils::inverseAxesMap(axesMap, rank);
    auto resetAxis = axesMap.size();
    for (auto &axis : inverseMap)
      if (axis < 0)
        axis = resetAxis++;
    llvm::interleaveComma(inverseMap, out);
    out << ",)";
  }

  void emitTCMulOp(MulOp op) {
    auto inputs = op.getInputs();
    auto result = op.getResult();
    auto axesMaps = op.getAxesMaps();
    auto rank = getTensorRank(result);
    SmallVector<std::string> inputNames;
    for (auto [position, input] : llvm::enumerate(inputs)) {
      int64_t iRank = axesMaps[position].size();
      std::string name = valueNameMap[input];
      if (iRank == 0) {
        inputNames.emplace_back(name);
        continue;
      }
      std::string termName = "term" + std::to_string(position);
      inputNames.emplace_back(termName);
      if (iRank != rank) {
        // term0 = relay.expand_dims(input, axis=1, num_newaxis=2)
        out.indent(indent) << termName << " = relay.expand_dims(" << name
                           << ", axis=" << iRank
                           << ", num_newaxis=" << (rank - iRank) << ")\n";
        name = termName;
      }
      out.indent(indent) << termName << " = relay.transpose(" << name
                         << ", axes=";
      emitInverseAxesMap(axesMaps[position], rank);
      out << ")\n";
    }
    std::string name = "var" + std::to_string(varIdx++);
    out.indent(indent) << name << " = relay.multiply(" << inputNames[0] << ", "
                       << inputNames[1] << ")\n";
    for (auto position : llvm::seq<int64_t>(2, inputs.size())) {
      out.indent(indent) << name << " = relay.multiply(" << name << ", "
                         << inputNames[position] << ")\n";
    }
    valueNameMap[result] = name;
  }

  void emitInverseAxesMap(AxesMapRefTy axesMap) {
    out << "(";
    auto inverseMap = utils::inverseAxesMap(axesMap);
    llvm::interleaveComma(inverseMap, out);
    out << ",)";
  }

  void emitTCAddOp(AddOp op) {
    auto inputs = op.getInputs();
    auto result = op.getResult();
    auto axesMaps = op.getAxesMaps();
    // term0 = relay.transpose(input, axes=(inverse_map))
    for (auto [position, input] : llvm::enumerate(inputs)) {
      out.indent(indent) << "term" << position << " = relay.transpose("
                         << valueNameMap[input] << ", axes=";
      emitInverseAxesMap(axesMaps[position]);
      out << ")\n";
    }
    std::string name = "var" + std::to_string(varIdx++);
    out.indent(indent) << name << " = relay.add(term0, term1)\n";
    for (auto position : llvm::seq<int64_t>(2, inputs.size())) {
      out.indent(indent) << name << " = relay.add(" << name << ", term"
                         << position << ")\n";
    }
    valueNameMap[result] = name;
  }

  void emitTCContractOp(ContractOp op) {
    auto input = op.getInput();
    auto result = op.getResult();
    auto axes = op.getAxes();
    std::string name = "var" + std::to_string(varIdx++);
    out.indent(indent) << name << " = relay.sum(" << valueNameMap[input]
                       << ", axis=[";
    llvm::interleaveComma(axes, out);
    out << "])\n";
    valueNameMap[result] = name;
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
    auto inputs = op.getOperands();
    if (inputs.size() != 1) {
      out.indent(indent) << "..... # only support one output\n";
      return;
    }
    out.indent(indent) << "func = relay.Function([";
    llvm::interleaveComma(inputTensorNames, out);
    out << "], " << valueNameMap[inputs[0]] << ")\n";
    out.indent(indent) << "mod = tvm.IRModule.from_expr(func)\n";
    out.indent(indent) << "return mod, shapes\n";
  }
};
} // namespace

namespace mlir::tc::codegen {
void emitTCToTVMRelay(ModuleOp op, llvm::raw_ostream &out) {
  SymbolTable symTable(op);
  auto funcOps = op.getOps<func::FuncOp>();
  for (auto funcOp : funcOps) {
    EmitTVMImpl(funcOp, out).emit();
  }
}
} // namespace mlir::tc::codegen
