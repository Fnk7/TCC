#ifndef TENSORCONTRACTION_PASSES_H
#define TENSORCONTRACTION_PASSES_H

#include <memory>

namespace mlir {
class Pass;
namespace contraction {
std::unique_ptr<mlir::Pass> createLowerToAffinePass();
std::unique_ptr<mlir::Pass> createReturnToArgumentPass();
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
} // namespace contraction
} // namespace mlir

#endif
