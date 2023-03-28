#include "llvm/Support/raw_ostream.h"

#include "TCC/ContractionTree.h"

using namespace llvm;

namespace tc {
const std::string SizeSymbol::implicitPrefix = "sz#";
const std::string IndexSymbol::implicitPrefix = "ci#";

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
  case NK_TreeTensor:
    return "TreeRef";
  case NK_InputTensor:
    return "InputRef";
  case NK_Negative:
    return "Negative";
  case NK_Constant:
    return "Constant";
  default:
    return "Undifined";
  }
}

static void dumpSizes(SizesTy &Sizes) {
  if (Sizes.size()) {
    errs() << "(";
    interleaveComma(Sizes, errs());
    errs() << ")";
  }
}
static void dumpIndexes(IndexUsesTy &IndexUses) {
  bool Comma = false;
  for (auto Index : IndexUses) {
    errs() << (Comma ? ", " : "(");
    if (auto PIndex = dyn_cast<ParamIndexSymbol>(Index)) {
      if (PIndex->isImplicit())
        errs() << PIndex->getValue();
      else
        errs() << PIndex->getName();
    } else if (auto IIndex = dyn_cast<IterIndexSymbol>(Index)) {
      errs() << "i#" << IIndex->getID();
    }
    Comma = true;
  }
  errs() << (Comma ? ")" : "");
}
static void dumpIndexes(IndexDefsTy &Indexes) {
  if (Indexes.size()) {
    errs() << "(i#" << Indexes[0]->getID();
    for (size_t i = 1; i < Indexes.size(); i++)
      errs() << ", i#" << Indexes[i]->getID();
    errs() << ")";
  }
}

void InputTensor::dump() {
  errs() << name;
  dumpSizes(shape);
}

void RootNode::dump(int indent) {
  errs().indent(indent);
  errs() << getKind() << " ";
  dumpSizes(Shape);
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
void TreeTensorNode::dump(int indent) {
  errs().indent(indent);
  errs() << TreeRef->getName();
  errs() << " [CTree] ";
  dumpIndexes(IndexUses);
  dumpChilds(indent);
}
void InputTensorNode::dump(int indent) {
  errs().indent(indent);
  errs() << InputRef->getName();
  errs() << " [Input] ";
  dumpIndexes(IndexUses);
  dumpChilds(indent);
}
void NegativeNode::dump(int indent) {
  errs().indent(indent);
  errs() << "Negative";
  dumpChilds(indent);
}
void AdditionNode::dump(int indent) {
  errs().indent(indent);
  errs() << "Addition [";
  for (auto Add : AddOrSub) {
    errs() << (Add ? "+" : "-") << ",";
  }
  errs() << "] {";
  dumpChilds(indent);
  errs().indent(indent) << "}\n";
}
void ConstantNode::dump(int indent) {
  errs().indent(indent);
  errs() << "Constant " << Value << "\n";
}

void Node::dump(int indent) {
  if (auto *RN = dyn_cast<RootNode>(this)) {
    RN->dump(indent);
  } else if (auto *CN = dyn_cast<ContractionNode>(this)) {
    CN->dump(indent);
  } else if (auto *N = dyn_cast<AdditionNode>(this)) {
    N->dump(indent);
  } else if (getKind() == NK_Multiplication) {
    errs().indent(indent);
    errs() << "Multiplication {";
    dumpChilds(indent);
    errs().indent(indent);
    errs() << "}\n";
  } else if (auto *TLN = dyn_cast<TreeTensorNode>(this)) {
    TLN->dump(indent);
  } else if (auto *ILN = dyn_cast<InputTensorNode>(this)) {
    ILN->dump(indent);
  } else if (auto N = dyn_cast<NegativeNode>(this)) {
    N->dump(indent);
  } else if (auto N = dyn_cast<ConstantNode>(this)) {
    N->dump(indent);
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

void SizeSymbol::dump() {
  errs() << this;
  if (getUpperBound()) {
    if (isConstant()) {
      errs() << " = " << getUpperBound();
    } else {
      errs() << " in (" << getLowerBound() << ", " << getUpperBound() << ")";
    }
  }
}

void SizeSymbol::dumpAliases() {
  bool Comma = false;
  errs() << this;
  auto Aliases = getInfo()->Aliases;
  for (auto Alias : Aliases) {
    if (Alias == this || Alias->isImplicit())
      continue;
    errs() << (Comma ? ", " : " alias (");
    errs() << Alias;
    Comma = true;
  }
  errs() << (Comma ? ")" : " none");
}

void ContractionForest::dump() {
  if (ArgSizes.size()) {
    bool Comma = false;
    for (auto Size : ArgSizes) {
      errs() << (Comma ? ", " : "Input sizes: ");
      Size->dump();
      Comma = true;
    }
    errs() << (Comma ? "\n" : "");
  }
  if (ConstSizes.size()) {
    bool Comma = false;
    for (auto Size : ConstSizes) {
      errs() << (Comma ? ", " : "Constant sizes: ");
      Size->dump();
      Comma = true;
    }
    errs() << (Comma ? "\n" : "");
  }
  if (ArgIndices.size()) {
    errs() << "Input indices: ";
    bool Comma = false;
    for (auto Index : ArgIndices) {
      if (Comma)
        errs() << ", ";
      errs() << Index->getName();
      Comma = true;
    }
    errs() << "\n";
  }
  if (ConstIndices.size()) {
    bool Comma = false;
    for (auto Index : ConstIndices) {
      if (Index->isImplicit())
        continue;
      errs() << (Comma ? ", " : "Constant indices: ");
      errs() << Index->getName() << " = " << Index->getValue();
      Comma = true;
    }
    errs() << (Comma ? "\n" : "");
  }
  errs() << "Input tensors: ";
  if (ArgTensors.size()) {
    ArgTensors[0]->dump();
    for (size_t i = 1; i < inputTensors.size(); i++) {
      errs() << ", ";
      ArgTensors[i]->dump();
    }
  }
  errs() << "\n";
  {
    bool Comma = false;
    for (auto Size : ArgSizes) {
      if (Size->isImplicit() || !Size->hasAliases())
        continue;
      errs() << (Comma ? ", " : "Sizes aliases: ");
      Size->dumpAliases();
      Comma = true;
    }
    errs() << (Comma ? "\n" : "");
  }
  errs() << "Contraction Trees:\n";
  for (auto &Tree : treeTensors)
    Tree->dump();
  errs() << "Result outputs: ";
  if (OutputCTrees.size()) {
    bool Comma = false;
    for (auto CT : OutputCTrees) {
      if (Comma)
        errs() << ", ";
      errs() << CT->getName();
      dumpSizes(CT->getShape());
      Comma = true;
    }
  }
  errs() << "\n";
  if (ReservedInputs.size()) {
    errs() << "Reserved inputs: ";
    bool Comma = false;
    for (auto RI : ReservedInputs) {
      if (Comma)
        errs() << ", ";
      errs() << RI->getName();
      Comma = true;
    }
    errs() << "\n";
  }
}
} // namespace tc
