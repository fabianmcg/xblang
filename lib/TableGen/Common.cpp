//===- Common.cpp - Common Tablegen classes ----------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines common tablegen classes.
//
//===----------------------------------------------------------------------===//

#include "xblang/TableGen/Common.h"

using namespace xblang::tablegen;

llvm::StringRef CppParameter::getCppType() const {
  if (const auto *init = llvm::dyn_cast<llvm::StringInit>(def))
    return init->getValue();
  const llvm::Record *record = cast<llvm::DefInit>(def)->getDef();
  return record->getValueAsString("type");
}

std::optional<llvm::StringRef> CppParameter::getDefaultValue() const {
  if (isa<llvm::StringInit>(def))
    return std::nullopt;
  const llvm::Record *record = cast<llvm::DefInit>(def)->getDef();
  std::optional<llvm::StringRef> value =
      record->getValueAsOptionalString("defaultValue");
  return value && !value->empty() ? value : std::nullopt;
}

bool CppParameter::isValid(const llvm::Init *def) {
  if (const auto *init = llvm::dyn_cast<llvm::StringInit>(def))
    return true;
  const llvm::Record *record = cast<llvm::DefInit>(def)->getDef();
  if (const llvm::RecordVal *type = record->getValue("type");
      !type || !type->getValue())
    return false;
  return true;
}
