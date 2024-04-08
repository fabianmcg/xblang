//===- Utils.h - Eval op utils -----------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares op evaluation utilities.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_EVAL_UTILS_H
#define XBLANG_EVAL_UTILS_H

#include "xblang/Dialect/XBLang/IR/Enums.h"
#include "xblang/Support/LLVM.h"

namespace xblang {
namespace eval {
/// Evaluates a binary operator. Both Lhs and Rhs must have the same type, and
/// be one of Int, Float, Index, or DenseArrays of the previous types.
Attribute evalBinOp(BinaryOperator op, Attribute lhs, Attribute rhs,
                    mlir::Builder &builder, Location loc);
/// Evaluates a cast operation. Both type and value must be one of Int, Float,
/// Index, or DenseArrays of the previous types.
Attribute evalCast(Type type, Attribute value, mlir::Builder &builder);
} // namespace eval
} // namespace xblang

#endif // XBLANG_EVAL_UTILS_H
