//===- Record.h - Record classes ----------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares common utilities for interacting with LLVM TableGen
// records.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_TABLEGEN_RECORD_H
#define XBLANG_TABLEGEN_RECORD_H

#include "xblang/Support/LLVM.h"
#include "llvm/TableGen/Record.h"
#include <string>
#include <vector>

namespace llvm {
class Record;
} // namespace llvm

namespace xblang {
namespace tablegen {
// Wrapper class for the Record TableGen class.
class Record {
public:
  explicit Record(const llvm::Record *def, llvm::StringRef classType);

  /// Returns whether the record is valid.
  explicit operator bool() const { return def != nullptr; }

  /// Returns the underlying def.
  const llvm::Record &getDef() const { return *def; }

  /// Returns the name of the def.
  llvm::StringRef getName() const { return def->getName(); }

  /// Returns the field value as a `RecordTy`, throwing an error if it not of
  /// `RecordTy` type.
  template <typename RecordTy>
  RecordTy getDef(StringRef field) const {
    assert(def && "null record.");
    return RecordTy(def->getValueAsDef(field));
  }

  /// Returns the field value as a vector of `RecordTy`, throwing an exception
  /// if the value is not a definition list.
  template <typename RecordTy>
  std::vector<RecordTy> getDefList(StringRef field) const {
    assert(def && "null record.");
    llvm::ListInit *list = def->getValueAsListInit(field);
    std::vector<RecordTy> defs;
    for (llvm::Init *init : list->getValues())
      if (auto def = getDefInitOrError(init, RecordTy::ClassType))
        defs.push_back(RecordTy(*def));
    return defs;
  }

  /// Returns the string field `extraClassDeclarations`.
  llvm::StringRef getExtraClassDeclarations() const;

  /// Returns the string field `extraClassDefinitions`.
  llvm::StringRef getExtraClassDefinitions() const;

  /// Return the record hold by defInit of class `classType`, throw an error if
  /// `init` is not a `DefInit` or is not of the correct class.
  std::optional<llvm::Record *>
  getDefInitOrError(llvm::Init *init, llvm::StringRef classType) const;

protected:
  const llvm::Record *def;
};

/// Helper class for defining Record subclasses.
template <typename RecordTy>
class RecordMixin : public Record {
public:
  using Base = RecordMixin;

  explicit RecordMixin(const llvm::Record *def)
      : Record(def, RecordTy::ClassType) {}

  /// Returns true if the record is of subclass `RecordTy`
  static bool isa(const llvm::Record *def) {
    return def && def->isSubClassOf(RecordTy::ClassType);
  }

  /// Returns the record as `RecordTy` or `std::nullopt` if it is not
  /// convertible.
  static std::optional<RecordTy> castOrNull(const llvm::Record *def) {
    if (isa(def))
      return RecordTy(def);
    return std::nullopt;
  }
};
} // namespace tablegen
} // namespace xblang

#endif // XBLANG_TABLEGEN_RECORD_H
