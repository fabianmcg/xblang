//===- Concept.cpp - TableGen Concept class ----------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Tablegen Concept class.
//
//===----------------------------------------------------------------------===//

#include "xblang/TableGen/Concept.h"
#include <stack>

using namespace xblang::tablegen;

Concept::Op Concept::getOp() const {
  Op op;
  getOp(op);
  return op;
}

void Concept::getOp(Op &op) const {
  Op::insertIns(op, def->getValueAsDag("args"));
}

void Concept::Op::insertIns(Op &op, llvm::DagInit *init) {
  for (unsigned i = 0; i < init->getNumArgs(); ++i) {
    const llvm::Record *def =
        llvm::cast<llvm::DefInit>(init->getArg(i))->getDef();
    StringRef name = init->getArgNameStr(i);
    if (def->isSubClassOf("Property")) {
      op.properties.push_back(
          mlir::tblgen::NamedProperty({name, mlir::tblgen::Property(def)}));
      continue;
    }
    mlir::tblgen::Constraint constraint(def);
    if (mlir::isa<mlir::tblgen::TypeConstraint>(constraint)) {
      mlir::tblgen::TypeConstraint type(def);
      op.operands.push_back(mlir::tblgen::NamedTypeConstraint({name, type}));
    } else if (mlir::isa<mlir::tblgen::Attribute>(constraint)) {
      mlir::tblgen::Attribute attr(def);
      op.attrs.push_back(mlir::tblgen::NamedAttribute({name, attr}));
    }
  }
}

llvm::SmallVector<Concept> xblang::tablegen::getAncestorConcepts(Concept cep) {
  llvm::SmallVector<Concept> concepts;
  std::stack<Concept> stack({cep});
  while (!stack.empty()) {
    cep = stack.top();
    stack.pop();
    llvm::ListInit *list = cep.getDef().getValueAsListInit("parentConcepts");
    for (llvm::Init *init : list->getValues())
      if (auto def = cep.getDefInitOrError(init, Concept::ClassType)) {
        concepts.push_back(Concept(*def));
        stack.push(Concept(*def));
      }
  }
  return concepts;
}
