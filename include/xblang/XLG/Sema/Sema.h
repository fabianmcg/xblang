//===- Sema.h - XLG semantic checks ------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares functions for registering semantic checker patterns.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_XLG_SEMA_SEMA_H
#define XBLANG_XLG_SEMA_SEMA_H

namespace mlir {
class DialectRegistry;
class MLIRContext;
} // namespace mlir

namespace xblang {
class GenericPatternSet;

namespace xlg {
/// Registers the XLG sema dialect extension in the given registry.
void registerXLGSemaInterface(mlir::DialectRegistry &registry);
/// Registers the XLG sema dialect extension in the given context.
void registerXLGSemaInterface(mlir::MLIRContext &context);
} // namespace xlg
} // namespace xblang

#endif // XBLANG_XLG_SEMA_SEMA_H
