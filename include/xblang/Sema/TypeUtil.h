//===- TypeUtil.h - Type utilities -------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares type utility function and classes.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_SEMA_TYPEUTIL_H
#define XBLANG_SEMA_TYPEUTIL_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "xblang/Support/LLVM.h"

namespace xblang {
/// Returns true if `type` is a fundamental type.
/// Primitive types:
/// - Signed integers: si8, si16, si32, si64, ...
/// - Unsigned integers: ui8, ui16, ui32, ui64, ...
/// - Sign-less integers: i8, i16, i32, i64, ...
/// - Floating-point: f16, f32, f64, f128, ...
/// - Other: Index
inline bool isPrimitiveType(Type type) {
  return type && type.isIntOrIndexOrFloat();
}

/// Returns true if `type` is an int type.
inline bool isIntType(Type type) { return isa<IntegerType>(type); }

/// Returns true if `type` is a signed int type.
inline bool isSignedIntType(Type type) {
  if (auto intTy = dyn_cast<IntegerType>(type))
    return intTy.isSigned();
  return false;
}

/// Returns true if `type` is an unsigned int type.
inline bool isUnsignedIntType(Type type) {
  if (auto intTy = dyn_cast<IntegerType>(type))
    return intTy.isUnsigned();
  return false;
}

/// Returns true if `type` is an signless int type.
inline bool isSignlessIntType(Type type) {
  if (auto intTy = dyn_cast<IntegerType>(type))
    return intTy.isSignless();
  return false;
}

/// Returns true if `type` is an index type.
inline bool isIndexType(Type type) { return type.isIndex(); }

/// Returns true if `type` is a float type.
inline bool isFloatType(Type type) { return isa<FloatType>(type); }

/// Promotes primitive types, so that operations can performed between them.
Type promotePrimitiveTypes(Type lhs, Type rhs);
} // namespace xblang

#endif // XBLANG_SEMA_TYPEUTIL_H
