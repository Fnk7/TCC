#ifndef TCC_PASSES_H
#define TCC_PASSES_H

#include <memory>

namespace mlir {
class Pass;
namespace tc {
std::unique_ptr<mlir::Pass> createLowerToAffinePass();
std::unique_ptr<mlir::Pass> createReturnToArgumentPass();
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
} // namespace tc
} // namespace mlir

#endif
