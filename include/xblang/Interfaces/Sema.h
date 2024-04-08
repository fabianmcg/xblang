//===- Sema.h - Semantic checking interfaces ---------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares semantic checking interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_INTREFACES_SEMA_H
#define XBLANG_INTREFACES_SEMA_H

#include "mlir/IR/DialectInterface.h"

namespace xblang {
class TypeSystem;
class GenericPatternSet;

/// A dialect interface for participating in semantic checks.
class SemaDialectInterface
    : public mlir::DialectInterface::Base<SemaDialectInterface> {
public:
  SemaDialectInterface(mlir::Dialect *dialect) : Base(dialect) {}

  /// Adds patterns to the pattern set.
  virtual void populateSemaPatterns(GenericPatternSet &set,
                                    TypeSystem &typeSystem) const = 0;
};
} // namespace xblang

#endif // XBLANG_INTREFACES_SEMA_H
