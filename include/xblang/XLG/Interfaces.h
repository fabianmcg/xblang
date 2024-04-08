//===- Interfaces.h - Base XBLang language interfaces ------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares base XBLang language interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_XLG_INTERFACES_H
#define XBLANG_XLG_INTERFACES_H

#include "mlir/IR/OpDefinition.h"

#include "xblang/XLG/XLGInterfaces.h.inc"

namespace xblang {
/// Returns the type attribute stored in the op.
inline mlir::Type getUnderlyingType(mlir::Operation *op) {
  if (auto iface = dyn_cast<TypeAttrInterface>(op))
    return iface.getType();
  return nullptr;
}

inline mlir::Type getUnderlyingType(mlir::Value value) {
  return value ? getUnderlyingType(value.getDefiningOp()) : nullptr;
}
} // namespace xblang

#endif // XBLANG_XLG_INTERFACES_H
