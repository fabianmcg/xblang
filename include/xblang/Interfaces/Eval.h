//===- Eval.h - Evaluation interfaces ----------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares op evaluation interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_INTREFACES_EVAL_H
#define XBLANG_INTREFACES_EVAL_H

#include "mlir/IR/DialectInterface.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class TypeConverter;
}

namespace xblang {
class GenericPatternSet;

/// A dialect interface for participating in evaluation of ops.
class EvalDialectInterface
    : public mlir::DialectInterface::Base<EvalDialectInterface> {
public:
  EvalDialectInterface(mlir::Dialect *dialect) : Base(dialect) {}

  /// Adds patterns to the pattern set.
  virtual void populateEvalPatterns(GenericPatternSet &patterns) const = 0;
};
} // namespace xblang

#endif // XBLANG_INTREFACES_EVAL_H
