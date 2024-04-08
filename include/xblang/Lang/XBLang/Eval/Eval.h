//===- Eval.h - XBG op evaluation --------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares function for registering op evaluation patterns.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_LANG_XBLANG_EVAL_EVAL_H
#define XBLANG_LANG_XBLANG_EVAL_EVAL_H

namespace mlir {
class DialectRegistry;
class MLIRContext;
} // namespace mlir

namespace xblang {
class GenericPatternSet;

namespace xbg {
/// Registers the XBG op eval dialect extension in the given registry.
void registerXBGEvalInterface(mlir::DialectRegistry &registry);
/// Registers the XBG op eval dialect extension in the given context.
void registerXBGEvalInterface(mlir::MLIRContext &context);

/// Populate XBG op eval patterns for declarations.
void populateDeclEvalPatterns(GenericPatternSet &patterns);

/// Populate XBG op eval patterns for expressions.
void populateExprEvalPatterns(GenericPatternSet &patterns);

/// Populate XBG op eval patterns for statements.
void populateStmtEvalPatterns(GenericPatternSet &patterns);

/// Populate XBG op eval patterns for types.
void populateTypeEvalPatterns(GenericPatternSet &patterns);
} // namespace xbg
} // namespace xblang

#endif // XBLANG_LANG_XBLANG_EVAL_EVAL_H
