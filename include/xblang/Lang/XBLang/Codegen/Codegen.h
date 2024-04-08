//===- Codegen.h - XBG code generation ---------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares function for registering code generation patterns.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_LANG_XBLANG_CODEGEN_CODEGEN_H
#define XBLANG_LANG_XBLANG_CODEGEN_CODEGEN_H

namespace mlir {
class DialectRegistry;
class MLIRContext;
class TypeConverter;
} // namespace mlir

namespace xblang {
class GenericPatternSet;

namespace xbg {
/// Registers the XBG code gen dialect extension in the given registry.
void registerXBGCGInterface(mlir::DialectRegistry &registry);
/// Registers the XBG code gen dialect extension in the given context.
void registerXBGCGInterface(mlir::MLIRContext &context);

/// Populate XBG code gen patterns for declarations.
void populateDeclCGPatterns(GenericPatternSet &patterns,
                            const mlir::TypeConverter *converter = nullptr);

/// Populate XBG code gen patterns for expressions.
void populateExprCGPatterns(GenericPatternSet &patterns,
                            const mlir::TypeConverter *converter = nullptr);

/// Populate XBG code gen patterns for statements.
void populateStmtCGPatterns(GenericPatternSet &patterns,
                            const mlir::TypeConverter *converter = nullptr);

/// Populate XBG code gen patterns for types.
void populateTypeCGPatterns(GenericPatternSet &patterns,
                            const mlir::TypeConverter *converter = nullptr);
} // namespace xbg
} // namespace xblang

#endif // XBLANG_LANG_XBLANG_CODEGEN_CODEGEN_H
