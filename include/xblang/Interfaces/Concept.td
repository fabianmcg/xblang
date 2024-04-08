//===- Concept.td - Concept MLIR interface  -------------------*- Tblgen-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the concept interface.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_INTERFACES_CONCEPT_TD
#define XBLANG_INTERFACES_CONCEPT_TD

include "mlir/IR/Interfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Concept interface
//===----------------------------------------------------------------------===//

def ConceptInterface : OpInterface<"OpConcept"> {
  let description = [{
    This interface describes an operation that defines a concept.
  }];
  let cppNamespace = "::xblang";
  let methods = [
    InterfaceMethod<"Returns the concept embedded in the op as a type.",
      "::mlir::Type", "getConceptType", (ins), [{}], [{
        return ::xblang::getConceptType($_op.getConceptClass());
      }]
    >,
    InterfaceMethod<"Returns the concept embedded in the op.",
      "::xblang::Concept*", "getOpConcept", (ins), [{}], [{
        return ::xblang::getConceptType($_op.getConceptClass()).getConceptClass().getConcept();
      }]
    >
  ];
}

#endif // XBLANG_INTERFACES_CONCEPT_TD
