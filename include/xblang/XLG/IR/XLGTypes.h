//===- XLGTypes.h - XLG dialect types  ---------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the XLG dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_XLG_IR_XLGTYPES_H
#define XBLANG_XLG_IR_XLGTYPES_H

#include "mlir/IR/OpDefinition.h"

#include "xblang/Basic/Concept.h"

#define GET_TYPEDEF_CLASSES
#include "xblang/XLG/IR/XLGOpsTypes.h.inc"

namespace xblang {
namespace xlg {
template <typename T>
bool isClass(mlir::Type type) {
  auto classTy = llvm::dyn_cast_or_null<ConceptType>(type);
  if (!classTy)
    return false;
  ConceptContainer c = classTy.getConceptClass();
  if (!c.getConcept())
    return false;
  return llvm::isa<T>(c.getConcept());
}

/// Returns an XLG class type with the specified concept.
ConceptType getConceptClass(Concept *cep);
} // namespace xlg
} // namespace xblang

#endif // XBLANG_XLG_IR_XLGTYPES_H
