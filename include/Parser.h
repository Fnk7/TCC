#ifndef TENSORCONTRACTION_PARSER_H
#define TENSORCONTRACTION_PARSER_H

#include "ContractionTree.h"

namespace contraction {
std::unique_ptr<ContractionForest> parseInputFile(StringRef FileName);
} // namespace contraction

#endif
