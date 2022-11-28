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

void RootNode::dump(int indent) {
  errs().indent(indent);
  errs() << TreeLeaf.lock()->getUniqueName() << " ";
  dumpSizes(Sizes);
  errs() << " -> ";
  dumpIndexes(IndexDefs);
  dumpChilds(indent);
}
void ContractionNode::dump(int indent) {
  errs().indent(indent);
  errs() << "Contraction ";
  dumpSizes(Sizes);
  errs() << " -> ";
  dumpIndexes(IndexDefs);
  dumpChilds(indent);
}
void TreeAsLeafNode::dump(int indent) {
  errs().indent(indent);
  errs() << TreeRef.lock()->getUniqueName();
  errs() << " [tree] ";
  dumpIndexes(IndexUses);
  dumpChilds(indent);
}
void InputAsLeafNode::dump(int indent) {
  errs().indent(indent);
  errs() << InputRef.lock()->getName();
  errs() << " [Input] ";
  dumpIndexes(IndexUses);
  dumpChilds(indent);
}

void Node::dump(int indent) {
  if (auto *RN = dyn_cast<RootNode>(this)) {
    RN->dump(indent);
  } else if (auto *CN = dyn_cast<ContractionNode>(this)) {
    CN->dump(indent);
  } else if (getKind() == NK_Addition) {
    errs().indent(indent);
    errs() << "Addition {";
    dumpChilds(indent);
    errs().indent(indent);
    errs() << "}\n";
  } else if (getKind() == NK_Multiplication) {
    errs().indent(indent);
    errs() << "Multiplication {";
    dumpChilds(indent);
    errs().indent(indent);
    errs() << "}\n";
  } else if (auto *TLN = dyn_cast<TreeAsLeafNode>(this)) {
    TLN->dump(indent);
  } else if (auto *ILN = dyn_cast<InputAsLeafNode>(this)) {
    ILN->dump(indent);
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
