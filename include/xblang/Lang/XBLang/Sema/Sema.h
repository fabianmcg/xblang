//===- Sema.h - XBG semantic checker -----------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares function for registering semantic patterns.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_LANG_XBLANG_SEMA_SEMA_H
#define XBLANG_LANG_XBLANG_SEMA_SEMA_H

namespace mlir {
class DialectRegistry;
class MLIRContext;
} // namespace mlir

namespace xblang {
class GenericPatternSet;

namespace xbg {
/// Registers the XBG sema dialect extension in the given registry.
void registerXBGSemaInterface(mlir::DialectRegistry &registry);
/// Registers the XBG sema dialect extension in the given context.
void registerXBGSemaInterface(mlir::MLIRContext &context);

/// Populate XBG semantic checks for declarations.
void populateDeclSemaPatterns(GenericPatternSet &set);

/// Populate XBG semantic checks for expressions.
void populateExprSemaPatterns(GenericPatternSet &set);

/// Populate XBG semantic checks for statements.
void populateStmtSemaPatterns(GenericPatternSet &set);

/// Populate XBG semantic checks for types.
void populateTypeSemaPatterns(GenericPatternSet &set);
} // namespace xbg
} // namespace xblang

#endif // XBLANG_LANG_XBLANG_SEMA_SEMA_H
