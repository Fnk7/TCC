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
std::unique_ptr<Pass> createConvertTCToTBLASPass(int64_t minContractSize = 8);
std::unique_ptr<Pass> createAddBatchPass();
std::unique_ptr<Pass> createSplitWorkloadPass();
std::unique_ptr<Pass> createTCDCEPass();
} // namespace tc
} // namespace mlir

#endif
