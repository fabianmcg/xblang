//===- Init.h - Initializes all the XLG components ---------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares initialization functions for XLG.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_XLG_INIT_H
#define XBLANG_XLG_INIT_H

#include "mlir/IR/DialectRegistry.h"
#include "xblang/XLG/Codegen/Codegen.h"
#include "xblang/XLG/IR/XLGDialect.h"
#include "xblang/XLG/Sema/Sema.h"

namespace xblang {
namespace xlg {
/// Registers XLG in the given registry.
inline void registerXLG(mlir::DialectRegistry &registry) {
  registry.insert<XLGDialect>();
  registerXLGCGInterface(registry);
  registerXLGSemaInterface(registry);
}

/// Registers XLG in the given context.
inline void registerXLG(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registerXLG(registry);
  context.appendDialectRegistry(registry);
}
} // namespace xlg
} // namespace xblang

#endif // XBLANG_XLG_INIT_H
