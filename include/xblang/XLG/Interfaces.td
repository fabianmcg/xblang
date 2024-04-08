//===- Interfaces.td - XLG Interfaces  ------------------------*- Tblgen-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines XLG interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_XLG_INTERFACES_TD
#define XBLANG_XLG_INTERFACES_TD

include "mlir/IR/Interfaces.td"

def TypeAttrInterface : OpInterface<"TypeAttrInterface"> {
  let description = [{
    This interface describes an operation that holds a `TypeAttr` named type.
  }];
  let cppNamespace = "::xblang";
  let methods = [
    InterfaceMethod<"Returns the type being held.",
      "::mlir::Type", "getType", (ins), [{
        return $_op.getUnderlyingType();
      }], [{}]
    >,
    InterfaceMethod<"Returns the type being held.",
      "::mlir::Type", "getUnderlyingType", (ins), [{}], [{
        if (auto attr = $_op.getTypeAttr())
          return attr.getValue();
        return nullptr;
      }]
    >
  ];
}

#endif // XBLANG_XLG_INTERFACES_TD
