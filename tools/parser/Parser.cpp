#include "llvm/Support/CommandLine.h"

#include "TCC/Parser.h"

using namespace llvm;

static cl::opt<std::string> InputFileName(cl::Positional,
                                          cl::desc("<Input contraction file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "TCC Parser\n");
  auto CF = contraction::parseInputFile(InputFileName);
  if (CF) {
    CF->dump();
    return 0;
  }
  return 1;
}
