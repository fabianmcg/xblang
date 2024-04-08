//===- Codegen.h - XLG code generation ---------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares functions for registering code generation patterns.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_XLG_CODEGEN_CODEGEN_H
#define XBLANG_XLG_CODEGEN_CODEGEN_H

namespace mlir {
class DialectRegistry;
class MLIRContext;
} // namespace mlir

namespace xblang {
class GenericPatternSet;

namespace xlg {
/// Registers the XLG code gen dialect extension in the given registry.
void registerXLGCGInterface(mlir::DialectRegistry &registry);
/// Registers the XLG code gen dialect extension in the given context.
void registerXLGCGInterface(mlir::MLIRContext &context);

/// Populate XLG code gen patterns for XLG.
void populateCGPatterns(GenericPatternSet &patterns);
} // namespace xlg
} // namespace xblang

#endif // XBLANG_XLG_CODEGEN_CODEGEN_H
