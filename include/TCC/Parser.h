#ifndef TCC_PARSER_H
#define TCC_PARSER_H

#include "TCC/ContractionTree.h"

namespace tc {
std::unique_ptr<ContractionForest> parseInputFile(StringRef FileName);
} // namespace tc

#endif
