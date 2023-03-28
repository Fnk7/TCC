#ifndef TCC_PARSER_H
#define TCC_PARSER_H

#include "llvm/ADT/StringRef.h"

#include "TCC/ContractionTree.h"

namespace tc {
std::unique_ptr<ContractionForest> parseInputFile(llvm::StringRef FileName);
} // namespace tc

#endif
