//===- ContextDialect.h - XB MLIR context ------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the XBLang compiler context.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_BASIC_CONTEXTDIALECT_H
#define XBLANG_BASIC_CONTEXTDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectInterface.h"
#include "xblang/Basic/Context.h"

namespace xblang {
/// Embed the XBLang context in the MLIR context as a dialect.
class XBContextDialect : public ::mlir::Dialect {
public:
  ~XBContextDialect() override;

  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("xb_context");
  }

  /// Returns the XBLang context.
  XBContext &getContext() { return ctx; }

private:
  explicit XBContextDialect(::mlir::MLIRContext *context);
  friend class ::mlir::MLIRContext;
  XBContext ctx;
};

/// A dialect interface for obtaining the XB context.
class XBContextDialectInterface
    : public mlir::DialectInterface::Base<XBContextDialectInterface> {
public:
  XBContextDialectInterface(mlir::Dialect *dialect) : Base(dialect) {}

  /// Returns the XBLang context.
  virtual XBContext *geXBContext();
};
} // namespace xblang

MLIR_DECLARE_EXPLICIT_TYPE_ID(::xblang::XBContextDialect)

#endif // XBLANG_BASIC_CONTEXTDIALECT_H
