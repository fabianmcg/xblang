//===- Record.cpp - Record classes -------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines common utilities for interacting with LLVM TableGen
// records.
//
//===----------------------------------------------------------------------===//

#include "xblang/TableGen/Record.h"
#include "llvm/TableGen/Error.h"

using namespace xblang::tablegen;

Record::Record(const llvm::Record *def, llvm::StringRef classType) : def(def) {
  assert((classType.empty() || def) &&
         "null records of an specific-class are invalid.");
  if (!classType.empty() && def && !def->isSubClassOf(classType))
    llvm::PrintFatalError(def->getLoc(),
                          "expected record of class type: " + classType +
                              ", but instead received: " + def->getName());
}

std::optional<llvm::Record *>
Record::getDefInitOrError(llvm::Init *init, llvm::StringRef classType) const {
  if (llvm::DefInit *defInit = dyn_cast<llvm::DefInit>(init);
      defInit && defInit->getDef()->isSubClassOf(classType))
    return defInit->getDef();
  llvm::PrintFatalError(
      def->getLoc(), "Init list is not entirely of the specified class type!");
  llvm_unreachable("PrintFatalError should abort the program!");
}

llvm::StringRef Record::getExtraClassDeclarations() const {
  if (!def || def->isValueUnset("extraClassDeclaration"))
    return "";
  return def->getValueAsString("extraClassDeclaration");
}

llvm::StringRef Record::getExtraClassDefinitions() const {
  if (!def || def->isValueUnset("extraClassDefinition"))
    return "";
  return def->getValueAsString("extraClassDefinition");
}
