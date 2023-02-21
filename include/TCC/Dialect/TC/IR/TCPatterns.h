#include "mlir/IR/PatternMatch.h"

namespace mlir::tc {
void getCanonicalizeForCSEPatterns(RewritePatternSet &patterns,
                                   MLIRContext *context);
}
