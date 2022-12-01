#include "llvm/Support/raw_ostream.h"

#include "TCC/ContractionTree.h"

using namespace llvm;

namespace contraction {
const std::string Node::getKindString(NodeKind K) {
  switch (K) {
  case NK_Root:
    return "Root";
  case NK_Contraction:
    return "Contraction";
  case NK_Addition:
    return "Addition";
  case NK_Multiplication:
    return "Multiplication";
  case NK_TreeAsLeaf:
    return "TreeRef";
  case NK_InputAsLeaf:
    return "InputRef";
  default:
    return "Undifined";
  }
}

static void dumpSizes(SizesTy &Sizes) {
  if (Sizes.size()) {
    errs() << "(Sz" << Sizes[0]->getUID();
    for (size_t i = 1; i < Sizes.size(); i++)
      errs() << ", Sz" << Sizes[i]->getUID();
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
  errs() << getKind() << " ";
  dumpSizes(Sizes);
  errs() << " -> ";
  dumpIndexes(IndexDefs);
  dumpChilds(indent);
}
void ContractionNode::dump(int indent) {
  errs().indent(indent);
  errs() << getKind() << " ";
  dumpSizes(Sizes);
  errs() << " -> ";
  dumpIndexes(IndexDefs);
  dumpChilds(indent);
}
void TreeAsLeafNode::dump(int indent) {
  errs().indent(indent);
  errs() << TreeRef->getName();
  errs() << " [CTree] ";
  dumpIndexes(IndexUses);
  dumpChilds(indent);
}
void InputAsLeafNode::dump(int indent) {
  errs().indent(indent);
  errs() << InputRef->getName();
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

void ContractionTree::dump() {
  errs() << "Tree " << getName() << "\n";
  Root->dump();
}

void ContractionForest::dump() {
  errs() << "Input Sizes: ";
  if (SizesMap.size()) {
    errs() << "Sz" << 0;
    for (size_t i = 1; i < SizesMap.size(); i++) {
      if (!SizesMap[i]->isFirstAppear())
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
  for (; i < SizesMap.size(); i++) {
    if (!SizesMap[i]->isFirstAppear()) {
      errs() << "Sizes relation: Sz" << i << " = Sz" << SizesMap[i]->getUID();
      break;
    }
  }
  if (i < SizesMap.size()) {
    for (i++; i < SizesMap.size(); i++) {
      if (!SizesMap[i]->isFirstAppear())
        errs() << ", Sz" << i << " = Sz" << SizesMap[i]->getUID();
    }
    errs() << "\n";
  }
  errs() << "Contraction Trees:\n";
  for (auto &Tree : Trees)
    Tree->dump();
  errs() << "Result outputs: ";
  if (OutputCTrees.size()) {
    errs() << OutputCTrees[0]->getName();
    for (size_t i = 1; i < OutputCTrees.size(); i++) {
      errs() << ", " << OutputCTrees[i]->getName();
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
