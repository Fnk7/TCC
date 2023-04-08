#include "mlir/IR/PatternMatch.h"

namespace mlir::tc {
void populatePermuteSimplifyPatterns(RewritePatternSet &patterns,
                                     MLIRContext *context);
void getCanonicalizeForCSEPatterns(RewritePatternSet &patterns,
                                   MLIRContext *context);
void populateSimpilfyCodeGenPatterns(RewritePatternSet &patterns,
                                     MLIRContext *context);
} // namespace mlir::tc
