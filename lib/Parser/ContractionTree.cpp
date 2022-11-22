#include "llvm/Support/raw_ostream.h"

#include "TCC/ContractionTree.h"

using namespace llvm;

namespace contraction {
static void dumpSizes(SizesTy &Sizes) {
  if (Sizes.size()) {
    errs() << "(Sz" << Sizes[0];
    for (size_t i = 1; i < Sizes.size(); i++)
      errs() << ", Sz" << Sizes[i];
    errs() << ")";
  }
}
static void dumpIndexes(IndexesTy &Indexes) {
  if (Indexes.size()) {
    errs() << "(i" << Indexes[0];
    for (size_t i = 1; i < Indexes.size(); i++)
      errs() << ", i" << Indexes[i];
    errs() << ")";
  }
}

void InputTensor::dump() {
  errs() << Name;
  dumpSizes(Sizes);
}

void Node::dump(int indent) {
  errs().indent(indent);
  if (Type == Root) {
    errs() << TreeLeaf.lock()->getUniqueName() << " ";
    dumpSizes(Sizes);
    errs() << " -> ";
    dumpIndexes(Indexes);
    dumpChilds(indent);
  } else if (Type == Addition) {
    errs() << "Addition {";
    dumpChilds(indent);
    errs().indent(indent);
    errs() << "}\n";
  } else if (Type == Multiplication) {
    errs() << "Multiplication {";
    dumpChilds(indent);
    errs().indent(indent);
    errs() << "}\n";
  } else if (Type == Contraction) {
    errs() << "Contraction ";
    dumpSizes(Sizes);
    errs() << " -> ";
    dumpIndexes(Indexes);
    dumpChilds(indent);
  } else if (Type == InputAsLeaf) {
    errs() << InputLeaf.lock()->getName();
    errs() << " [Input] ";
    dumpIndexes(Indexes);
    dumpChilds(indent);
  } else if (Type == TreeAsLeaf) {
    errs() << TreeLeaf.lock()->getUniqueName();
    errs() << " [tree] ";
    dumpIndexes(Indexes);
    dumpChilds(indent);
  }
}
void Node::dumpChilds(int indent) {
  errs() << "\n";
  for (auto &C : Childs)
    C->dump(indent + 2);
}

void ContractionForest::dump() {
  errs() << "Input Sizes: ";
  if (Sizes.size()) {
    errs() << "Sz" << 0;
    for (size_t i = 1; i < Sizes.size(); i++) {
      if ((int)i == Sizes[i])
        errs() << ", Sz" << i;
    }
  }
  errs() << "\n";
  errs() << "Input tensors: ";
  if (InputTensors.size()) {
    InputTensors[0]->dump();
    for (size_t i = 1; i < InputTensors.size(); i++) {
      errs() << ", ";
      InputTensors[i]->dump();
    }
  }
  errs() << "\n";
  errs() << "Input size map: ";
  for (size_t i = 0; i < InputSizeMap.size(); i++) {
    if (i != 0)
      errs() << ", ";
    errs() << InputSizeMap[i];
  }
  errs() << "\n";
  size_t i = 0;
  for (; i < Sizes.size(); i++) {
    if ((int)i != Sizes[i]) {
      errs() << "Sizes relation: Sz" << i << " = Sz" << Sizes[i];
      break;
    }
  }
  if (i < Sizes.size()) {
    for (i++; i < Sizes.size(); i++) {
      if ((int)i != Sizes[i])
        errs() << ", Sz" << i << " = Sz" << Sizes[i];
    }
    errs() << "\n";
  }
  errs() << "Contraction Trees:\n";
  for (auto &tree : Trees)
    tree->dump();
  errs() << "Result outputs: ";
  if (OutputTrees.size()) {
    errs() << OutputTrees[0]->getUniqueName();
    for (size_t i = 1; i < OutputTrees.size(); i++) {
      errs() << ", " << OutputTrees[i]->getUniqueName();
    }
  }
  errs() << "\n";
  if (ReservedInputs.size()) {
    errs() << "Reserved inputs: ";
    errs() << ReservedInputs[0]->getName();
    for (size_t i = 1; i < ReservedInputs.size(); i++) {
      errs() << ", " << ReservedInputs[i]->getName();
    }
    errs() << "\n";
  }
}
} // namespace contraction
