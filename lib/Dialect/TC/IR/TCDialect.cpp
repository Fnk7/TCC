#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "TCC/Dialect/TC/IR/TCOps.h"

using namespace mlir;
using namespace mlir::tc;

#include "TCC/Dialect/TC/IR/TCOpsDialect.cpp.inc"

void TCDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TCC/Dialect/TC/IR/TCOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "TCC/Dialect/TC/IR/TCAttributes.cpp.inc"
      >();
}

#define GET_ATTRDEF_CLASSES
#include "TCC/Dialect/TC/IR/TCAttributes.cpp.inc"

void AxesAttr::print(AsmPrinter &printer) const {
  auto axesRef = getAxes();
  printer << "[";
  llvm::interleaveComma(axesRef, printer);
  printer << "]";
}

Attribute AxesAttr::parse(AsmParser &parser, Type odsType) {
  AxesTy axes;
  assert(axes.size() && "Error: Need to custom parser for AxesAttr.");
  return AxesAttr::get(odsType.getContext(), axes);
}

void AxesMapsAttr::print(AsmPrinter &printer) const {
  auto axesMapsRef = getAxesMaps();
  printer << "[";
  llvm::interleaveComma(axesMapsRef, printer, [&](AxesMapRefTy axesMapRef) {
    printer << "(";
    llvm::interleaveComma(axesMapRef, printer);
    printer << ")";
  });
  printer << "]";
}

Attribute AxesMapsAttr::parse(AsmParser &parser, Type odsType) {
  AxesMapsTy axesMaps;
  assert(axesMaps.size() && "Error: Need to custom parser for AxesMapsAttr.");
  return AxesMapsAttr::get(odsType.getContext(), axesMaps);
}

void SignsAttr::print(AsmPrinter &printer) const {
  auto signsRef = getSigns();
  printer << "[";
  llvm::interleaveComma(signsRef, printer, [&](bool sign) {
    if (sign) {
      printer << "+";
    } else {
      printer << "-";
    }
  });
  printer << "]";
}

Attribute SignsAttr::parse(AsmParser &parser, Type odsType) {
  SignsTy signs;
  assert(signs.size() && "Error: Need to custom parser for SignsAttr.");
  return SignsAttr::get(odsType.getContext(), signs);
}

void SymmetryAttr::print(AsmPrinter &printer) const {
  auto symmetryRef = getSymmetry();
  if (symmetryRef.size()) {
    printer << "<";
    llvm::interleaveComma(symmetryRef, printer, [&](SymmPairTy symm) {
      printer << symm.first;
      printer << "~";
      printer << symm.second;
    });
    printer << ">";
  }
}

Attribute SymmetryAttr::parse(AsmParser &parser, Type odsType) {
  SymmArrayTy symmetry;
  assert(symmetry.size() && "Error: Need to custom parser for SymmetryAttr.");
  return SymmetryAttr::get(odsType.getContext(), symmetry);
}
