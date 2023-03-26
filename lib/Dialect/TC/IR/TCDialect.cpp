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
  printer << "<";
  llvm::interleaveComma(axesRef, printer);
  printer << ">";
}

Attribute AxesAttr::parse(AsmParser &parser, Type odsType) {
  AxesTy axes;
  return AxesAttr::get(odsType.getContext(), axes);
}

void AxesMapAttr::print(AsmPrinter &printer) const {
  auto axesMapRef = getAxesMap();
  printer << "<";
  llvm::interleaveComma(axesMapRef, printer);
  printer << ">";
}

Attribute AxesMapAttr::parse(AsmParser &parser, Type odsType) {
  AxesMapTy axesMap;
  return AxesMapAttr::get(odsType.getContext(), axesMap);
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
  return AxesMapsAttr::get(odsType.getContext(), axesMaps);
}
