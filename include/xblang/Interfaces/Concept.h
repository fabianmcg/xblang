//===- Concept.h - Concept interface ----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines concept related classes and interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_INTERFACES_CONCEPT_H
#define XBLANG_INTERFACES_CONCEPT_H

#include "mlir/IR/OpDefinition.h"
#include "xblang/Basic/Concept.h"

namespace xblang {
namespace xlg {
class ConceptType;
}

/// Class for obtaining the underlying concept behind a type.
template <typename T>
struct ConceptTypeTraits {
  using type = void;

  template <typename ConceptType>
  static void get(ConceptType val) {
    static_assert(false);
  }
};

template <>
struct ConceptTypeTraits<xlg::ConceptType> {
  using type = xlg::ConceptType;

  template <typename ConceptType>
  static ConceptType get(ConceptType type) {
    return type;
  }
};

template <>
struct ConceptTypeTraits<mlir::TypedValue<xlg::ConceptType>> {
  using type = xlg::ConceptType;

  template <typename ConceptType = type>
  static ConceptType get(mlir::TypedValue<ConceptType> val) {
    return val.getType();
  }
};

/// Returns the XLG concept type inside type.
template <typename ConceptType>
ConceptTypeTraits<ConceptType>::type getConceptType(ConceptType type) {
  return ConceptTypeTraits<ConceptType>::get(type);
}
} // namespace xblang

#include "xblang/Interfaces/ConceptInterfaces.h.inc"

namespace xblang {
/// Return the closest surrounding parent operation that is of concept
/// 'Concept'.
template <typename Concept>
OpConcept getParentOfConcept(mlir::Operation *op) {
  if (!op)
    return nullptr;
  OpConcept parent = op->getParentOfType<OpConcept>();
  while (parent) {
    if (llvm::isa<Concept>(parent.getOpConcept()))
      break;
    parent = parent->getParentOfType<OpConcept>();
  }
  return parent;
}

/// Class for concept interfaces.
template <typename Concept>
class ConceptInterface {
public:
  using ConceptType = Concept;
  using Interface = typename Concept::PureInterface;
  ConceptInterface() = default;

  ConceptInterface(mlir::Operation *op) : op(op) {
    ::xblang::Concept *cep{};
    if (auto cepOp = dyn_cast<OpConcept>(op);
        cepOp && (cep = cepOp.getOpConcept()))
      interface = reinterpret_cast<Interface *>(cep->getRawInterface(
          op->getName().getTypeID(), TypeInfo::get<Concept>()));
  }

  operator bool() const { return interface && op; }

  /// Returns whether a concept has certain class.
  static inline bool classof(mlir::Operation const *op) {
    if (auto cepOp = dyn_cast<OpConcept>(op))
      return mlir::isa<Concept>(cepOp.getOpConcept());
    return false;
  }

  mlir::Operation *getOp() const { return op; }

protected:
  ConceptInterface(Interface *iface, mlir::Operation *op)
      : interface(iface), op(op) {}

  Interface *getImpl() const { return interface; }

  /// An instance of the construct.
  Interface *interface{};
  /// The op using the concept.
  mlir::Operation *op{};
};
} // namespace xblang

#endif // XBLANG_INTERFACES_CONCEPT_H
