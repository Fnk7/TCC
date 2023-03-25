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
    /// TVM auto_scheduler
    out << "@auto_scheduler.register_workload\n";
    out << "def " << funcOp.getName() << "(";
    emitInputSizes();
    out << "):\n";
    indent += 4;
    emitInputTensors();
    emitFuncionBody();
  }

private:
  int indent = 0;
  using ValueNameMapTy = DenseMap<Value, std::string>;
  ValueNameMapTy valueNameMap;

  void emitInputSizes() {
    for (auto idx : llvm::seq<int>(0, numInputSizes)) {
      if (idx != 0) {
        out << ", ";
      }
      out << "s" << idx;
    }
  }

  void emitInputTensors() {
    for (auto argValue : funcOp.getArguments()) {
      if (argValue.getType().isa<RankedTensorType>()) {
        std::string name = "t" + std::to_string(argValue.getArgNumber());
        out.indent(indent) << name << " = te.placeholder(";
        emitTensorSizes(argValue);
        out << ", name=" << name << ")\n";
        valueNameMap[argValue] = name;
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
      if (size >= 0) {
        out << size;
      } else {
        out << "s" << sizeIDs[idx];
      }
    }
    out << ")";
  }

  void emitFuncionBody() {
    for (auto &op : block) {
      if (auto addOp = dyn_cast_or_null<AddOp>(&op)) {
        out.indent(indent) << "# TODO: " << op.getName() << "\n";
      } else if (auto mulOp = dyn_cast_or_null<MulOp>(&op)) {
        emitMulOp(mulOp);
      } else {
        out.indent(indent) << "# TODO: " << op.getName() << "\n";
      }
    }
  }

  template <typename Op> void emitTCAddOrMulOp(Op op) {}

  void emitMulOp(MulOp op) {
    auto inputs = op.getInputs();
    auto axesMaps = op.getAxesMapsArray();
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
