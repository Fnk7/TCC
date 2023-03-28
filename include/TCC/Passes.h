#ifndef TCC_PASSES_H
#define TCC_PASSES_H

#include <memory>

namespace mlir {
class Pass;
namespace tc {
std::unique_ptr<Pass> createLowerToAffinePass();
std::unique_ptr<Pass> createReturnToArgumentPass();
std::unique_ptr<Pass> createLowerToLLVMPass();
std::unique_ptr<Pass>
createConvertTCToLinalgPass(bool eraseTensorEncoding = false);
std::unique_ptr<Pass> createConvertTCToTBLASPass();
} // namespace tc
} // namespace mlir

#endif
