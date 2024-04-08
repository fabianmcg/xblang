//===- Syntax.h - Syntax interfaces ------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares syntax related interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_INTREFACES_SYNTAX_H
#define XBLANG_INTREFACES_SYNTAX_H

#include "mlir/IR/DialectInterface.h"

namespace xblang {
class SyntaxContext;
class XBContext;

/// A dialect interface for participating in syntax parsing.
class SyntaxDialectInterface
    : public mlir::DialectInterface::Base<SyntaxDialectInterface> {
public:
  SyntaxDialectInterface(mlir::Dialect *dialect) : Base(dialect) {}

  /// Add combinators to the dynamic parser.
  virtual void populateSyntax(XBContext *context,
                              SyntaxContext &syntaxContext) const = 0;
};
} // namespace xblang

#endif // XBLANG_INTREFACES_SYNTAX_H
