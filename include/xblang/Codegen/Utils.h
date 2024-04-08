//===- Utils.h - Common builder utilities ------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares common builder utility functions and classes.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_CODEGEN_UTILS_H
#define XBLANG_CODEGEN_UTILS_H

#include "mlir/IR/Value.h"
#include "xblang/Dialect/XBLang/IR/Enums.h"

namespace mlir {
class OpBuilder;
}

namespace xblang {
class CastInfo;
/// Creates a binary expression using standard MLIR dialects. Type is used to
/// indicate the signedness of the operation. It returns null if the operands
/// are invalid.
mlir::Value createArithBinOp(mlir::OpBuilder &builder, BinaryOperator op,
                             mlir::Value lhs, mlir::Value rhs,
                             mlir::Type type = nullptr);
/// Creates a cast expression using standard MLIR dialects. It returns null if
/// the operands are invalid.
mlir::Value createCastOp(mlir::Type dstType, mlir::Type srcType,
                         mlir::Value expr, mlir::OpBuilder &builder,
                         CastInfo *converter);
} // namespace xblang

#endif // XBLANG_CODEGEN_UTILS_H
