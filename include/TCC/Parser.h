#ifndef TCC_PARSER_H
#define TCC_PARSER_H

#include "llvm/ADT/StringRef.h"

namespace tc {
class ContractionForest;
std::unique_ptr<ContractionForest> parseInputFile(llvm::StringRef FileName);
} // namespace tc

#endif
