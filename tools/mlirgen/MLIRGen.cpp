#include "llvm/Support/CommandLine.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "Parser.h"

#include "MLIRGen.h"
#include "TCDialect/ContractionOps.h"

using namespace llvm;

static cl::opt<std::string> InputFileName(cl::Positional,
                                          cl::desc("<Input contraction file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");
  auto CF = contraction::parseInputFile(InputFileName);
  if (!CF)
    return 1;
  mlir::MLIRContext Ctx;
  Ctx.getOrLoadDialect<mlir::scf::SCFDialect>();
  Ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  Ctx.getOrLoadDialect<mlir::contraction::ContractionDialect>();
  auto M = mlir::contraction::MLIRGen(Ctx, *CF);
  M->dump();
  return 0;
}
