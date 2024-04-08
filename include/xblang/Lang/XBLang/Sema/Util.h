//===- Util.h - Common semantic utilities ------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares common semantic utility functions.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_LANG_XBLANG_SEMA_UTIL_H
#define XBLANG_LANG_XBLANG_SEMA_UTIL_H

#include "mlir/IR/Value.h"
#include "xblang/Support/LLVM.h"

namespace xblang {
namespace sema {
class SemaDriver;
}

namespace xbg {
/// Helper struct for passing augmented values.
struct Operand {
  Operand(Value value, Type type) : value(value), type(type) {}

  operator bool() const { return value && type; }

  operator Value() const { return value; }

  operator Type() const { return type; }

  Operand &operator=(Value val) {
    value = val;
    return *this;
  }

  Operand &operator=(Type ty) {
    type = ty;
    return *this;
  }

  Value value;
  Type type;
};

/// Returns the underlying type of a reference, pointer or memref, or nullptr if
/// it's not a dereferenceable type.
Type getElementType(Type type);
/// Returns the underlying type of a reference, pointer or memref, or the type
/// if it's not a dereferenceable type.
Type getTypeOrElementType(Type type);
/// Loads a value from a reference or pointer and returns the loaded value, or
/// returns the value itself if it's already a pure value.
Operand getOrLoadValue(Operand operand, sema::SemaDriver &driver);
/// Emits a store operation, returns nullptr in failure.
Operation *storeValue(Operand address, Operand value, sema::SemaDriver &driver);
} // namespace xbg
} // namespace xblang

#endif // XBLANG_LANG_XBLANG_SEMA_UTIL_H
