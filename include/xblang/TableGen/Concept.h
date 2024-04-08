//===- Concept.h - TableGen Concept class ------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Tablegen Concept class.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_TABLEGEN_CONCEPT_H
#define XBLANG_TABLEGEN_CONCEPT_H

#include "mlir/TableGen/Argument.h"
#include "mlir/TableGen/Region.h"
#include "xblang/TableGen/Record.h"

namespace llvm {
class Record;
} // namespace llvm

namespace xblang {
namespace tablegen {
//===----------------------------------------------------------------------===//
// Concept
//===----------------------------------------------------------------------===//
// Wrapper class for the Concept TableGen class.
class Concept : public RecordMixin<Concept> {
public:
  static constexpr std::string_view ClassType = "Concept";
  using Base::Base;

  // Returns the C++ class name.
  StringRef getClassName() const { return def->getValueAsString("className"); }

  // Returns the concept's mnemonic.
  StringRef getMnemonic() const { return def->getValueAsString("conceptName"); }

  // Returns the concept's dialect mnemonic.
  StringRef getDialectMnemonic() const {
    return def->isValueUnset("dialectName")
               ? ""
               : def->getValueAsString("dialectName");
  }

  // Returns the cpp namespace.
  StringRef getCppNamespace() const {
    return def->getValueAsString("cppNamespace");
  }

  // Returns the parent concepts.
  auto getParentConcepts() const {
    return getDefList<Concept>("parentConcepts");
  }

  /// Returns whether the concept is a pure construct.
  bool getPureConstruct() const { return def->getValueAsBit("pureConstruct"); }

  /// Struct for representing the MLIR constructs specified by the concept.
  struct Op {
    /// Inserts the Op ins field from `init`.
    static void insertIns(Op &op, llvm::DagInit *init);
    /// Inserts the Op outs field from `init`.
    static void insertOuts(Op &op, llvm::DagInit *init);
    /// Inserts the Op regions field from `init`.
    static void insertRegions(Op &op, llvm::DagInit *init);
    /// MLIR Op ins
    llvm::SmallVector<mlir::tblgen::NamedAttribute, 4> attrs;
    llvm::SmallVector<mlir::tblgen::NamedProperty, 4> properties;
    llvm::SmallVector<mlir::tblgen::NamedTypeConstraint, 4> operands;
    /// MLIR Op outs
    llvm::SmallVector<mlir::tblgen::NamedTypeConstraint, 4> outs;
    /// MLIR Op regions
    llvm::SmallVector<mlir::tblgen::NamedRegion, 4> regions;
  };

  /// Returns the Op constructs specified by the concept.
  Op getOp() const;
  void getOp(Op &op) const;
};

// Returns all the ancestor concepts.
llvm::SmallVector<Concept> getAncestorConcepts(Concept cep);

//===----------------------------------------------------------------------===//
// Construct
//===----------------------------------------------------------------------===//
// Wrapper class for the Construct TableGen class.
class Construct : public RecordMixin<Construct> {
public:
  static constexpr std::string_view ClassType = "Construct";
  using Base::Base;
};
} // namespace tablegen
} // namespace xblang

#endif // XBLANG_TABLEGEN_CONCEPT_H
