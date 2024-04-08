//===- Traits.h - XLG traits -------------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares XLG MLIR traits.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_XLG_IR_TRAITS_H
#define XBLANG_XLG_IR_TRAITS_H

#include "xblang/XLG/IR/XLGTypes.h"
#include "xblang/XLG/Interfaces.h"

namespace xblang {
/// Returns an XB context stored in the dialect.
XBContext *getXBContext(mlir::Dialect *dialect);
XBContext *getXBContext(mlir::MLIRContext *context);

namespace xlg {
/// Parses an XLG concept type.
mlir::ParseResult parseXLGConcept(mlir::OpAsmParser &parser, mlir::Type &type,
                                  Concept *cep);
/// Prints an XLG concept type.
void printXLGConcept(mlir::OpAsmPrinter &printer, ConceptType type,
                     Concept *cep);
/// Parses an XLG concept type.
mlir::ParseResult parseXLGConcept(mlir::OpAsmParser &parser,
                                  mlir::TypeAttr &type, Concept *cep);
/// Prints an XLG concept type.
void printXLGConcept(mlir::OpAsmPrinter &printer, mlir::TypeAttr type,
                     Concept *cep);

/// XLG concept MLIR trait.
template <typename ConcreteType>
class ConceptTrait
    : public ::mlir::OpTrait::TraitBase<ConcreteType, ConceptTrait> {
public:
  /// Returns the XB context.
  XBContext *getXBContext() {
    return ::xblang::getXBContext((*this).getOperation()->getDialect());
  }

  /// Returns the default class type for the concept.
  ConceptType getXLGClass() {
    return getConceptClass(
        static_cast<ConcreteType &>(*this).getConcept(getXBContext()));
  }

  /// Parses an XLG concept type.
  static mlir::ParseResult parseConcept(mlir::OpAsmParser &parser,
                                        mlir::Type &type) {
    return parseXLGConcept(
        parser, type,
        ConcreteType::getConcept(::xblang::getXBContext(parser.getContext())));
  }

  /// Parses an XLG concept type.
  static void printConcept(mlir::OpAsmPrinter &printer, ConcreteType op,
                           ConceptType type) {
    printXLGConcept(
        printer, type,
        ConcreteType::getConcept(::xblang::getXBContext(op.getContext())));
  }

  /// Parses an XLG concept type.
  static mlir::ParseResult parseConcept(mlir::OpAsmParser &parser,
                                        mlir::TypeAttr &type) {
    return parseXLGConcept(
        parser, type,
        ConcreteType::getConcept(::xblang::getXBContext(parser.getContext())));
  }

  /// Parses an XLG concept type.
  static void printConcept(mlir::OpAsmPrinter &printer, ConcreteType op,
                           mlir::TypeAttr type) {
    printXLGConcept(
        printer, type,
        ConcreteType::getConcept(::xblang::getXBContext(op.getContext())));
  }
};
} // namespace xlg
} // namespace xblang

#endif // XBLANG_XLG_IR_TRAITS_H
