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
  printer << "<(";
  llvm::interleaveComma(axesRef, printer);
  printer << ")>";
}

Attribute AxesAttr::parse(AsmParser &parser, Type odsType) {
  AxesTy axes;
  assert(axes.size() && "TODO: Custom parser for AxesAttr.");
  return AxesAttr::get(odsType.getContext(), axes);
}

void AxesMapsAttr::print(AsmPrinter &printer) const {
  auto axesMapsRef = getAxesMaps();
  printer << "<";
  llvm::interleaveComma(axesMapsRef, printer, [&](AxesMapRefTy axesMapRef) {
    printer << "(";
    llvm::interleaveComma(axesMapRef, printer);
    printer << ")";
  });
  printer << ">";
}

Attribute AxesMapsAttr::parse(AsmParser &parser, Type odsType) {
  AxesMapsTy axesMaps;
  assert(axesMaps.size() && "TODO: Custom parser for AxesAttr.");
  return AxesMapsAttr::get(odsType.getContext(), axesMaps);
}

void SignsAttr::print(AsmPrinter &printer) const {
  auto signsRef = getSigns();
  printer << "<";
  llvm::interleaveComma(signsRef, printer, [&](bool sign) {
    if (sign) {
      printer << "+";
    } else {
      printer << "-";
    }
  });
  printer << ">";
}

Attribute SignsAttr::parse(AsmParser &parser, Type odsType) {
  SignsTy signs;
  assert(signs.size() && "TODO: Custom parser for AxesAttr.");
  return SignsAttr::get(odsType.getContext(), signs);
}
