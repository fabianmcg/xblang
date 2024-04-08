//===- Concept.h - Language concept  -----------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the language concept class.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_BASIC_CONCEPT_H
#define XBLANG_BASIC_CONCEPT_H

#include "xblang/Basic/TypeInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"
#include <tuple>

namespace mlir {
class MLIRContext;
class Operation;
}

namespace xblang {
class XBContext;
class Concept;

/// Base class for all concepts.
class Concept {
public:
  // Make the class abstract.
  virtual ~Concept() = 0;

  /// Returns whether a concept has certain class.
  static inline bool classof(Concept const *c) { return true; }

  /// Returns the XBLang context owning this concept.
  XBContext *getContext() const { return context; }

  /// Returns the MLIR context in this concept.
  mlir::MLIRContext *getMLIRContext() const;

  /// Returns the MLIR context in the concept.
  static mlir::MLIRContext *getMLIRContext(Concept *cep) {
    return cep ? cep->getMLIRContext() : nullptr;
  }

  /// Returns the concept dialect.
  llvm::StringRef getDialect() const { return dialect; }

  /// Returns the concept identifier.
  llvm::StringRef getIdentifier() const { return identifier; }

  /// Returns the type info.
  TypeInfo getTypeInfo() const { return typeInfo; }

  /// Returns whether is first class concept.
  bool isFirstClass() const { return dialect.empty(); }

  /// Returns the pointer to the raw interface.
  void *getRawInterface(mlir::TypeID constructID, TypeInfo info);

  void *getRawInterface(mlir::TypeID constructID) {
    return getRawInterface(constructID, typeInfo);
  }

protected:
  friend class XBContext;
  /// Get the concept stored in the context given the TypeInfo.
  static Concept *get(XBContext *context, TypeInfo typeInfo);

  Concept(XBContext *context, llvm::StringRef identifier,
          llvm::StringRef dialect, TypeInfo typeInfo)
      : context(context), dialect(dialect), identifier(identifier),
        typeInfo(typeInfo) {}

  static bool isClassof(TypeInfo info, Concept const *c);
  /// XBLang context.
  XBContext *context;
  /// Dialect identifier.
  llvm::StringRef dialect;
  /// Concept identifier.
  llvm::StringRef identifier;
  /// Type identifier for the concept.
  TypeInfo typeInfo;
};

/// Class for holding concepts.
class ConceptContainer {
public:
  ConceptContainer(Concept *c) : c(c) {}

  operator bool() const { return c != nullptr; }

  /// Returns the xblang context.
  XBContext *getContext() const {
    if (c)
      return c->getContext();
    return nullptr;
  }

  /// Returns the concept dialect.
  llvm::StringRef getDialect() const {
    if (c)
      return c->getDialect();
    return "";
  }

  /// Returns the concept identifier.
  llvm::StringRef getIdentifier() const {
    if (c)
      return c->getIdentifier();
    return "";
  }

  /// Compare two concepts using their addresses.
  bool operator==(const ConceptContainer &other) const { return c == other.c; }

  /// Hash concepts according to the info pointer.
  llvm::hash_code hash() const { return llvm::hash_value(c); }

  /// Returns the concept.
  Concept *getConcept() const { return c; }

private:
  Concept *c;
};

/// Hash a ConceptInfo value
inline llvm::hash_code hash_value(const ConceptContainer &value) {
  return value.hash();
}

template <typename ConcreteTy, typename... ParentConceptsTy>
class ConceptMixin : public Concept {
public:
  using ConcreteConcept = ConcreteTy;
  using ParentConcepts = std::tuple<ParentConceptsTy...>;
  using Base = ConceptMixin;
  ~ConceptMixin() = default;

  /// Returns the Concept Mnemonic.
  static llvm::StringRef getMnemonic() { return ConcreteTy::mnemonic; }

  /// Returns the Dialect Mnemonic.
  static llvm::StringRef getDialectMnemonic() {
    return ConcreteTy::dialect_mnemonic;
  }

  /// Returns whether a concept has certain class.
  static inline bool classof(Concept const *c) {
    return isClassof(TypeInfo::get<ConcreteTy>(), c);
  }

  /// Returns the type info of the concept.
  static TypeInfo getTypeInfo() { return TypeInfo::get<ConcreteConcept>(); }

  /// Get the concept stored in the context given the TypeInfo.
  static ConcreteTy *get(XBContext *context) {
    return static_cast<ConcreteTy *>(Concept::get(context, getTypeInfo()));
  }

  /// Registers the parent concepts.
  void initialize(XBContext *context) {}

protected:
  friend class XBContext;

  ConceptMixin(XBContext *context)
      : Concept(context, getMnemonic(), getDialectMnemonic(), getTypeInfo()) {
    static_cast<ConcreteTy &>(*this).initialize(context);
  }
};

/// Base class for concept models.
struct ModelInterface {
  virtual ~ModelInterface() = default;
};
} // namespace xblang

#endif // XBLANG_BASIC_CONCEPT_H
