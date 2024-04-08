//===- Init.h - Initializes all the Par components ---------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares initialization functions for par.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_LANG_PARALLEL_INIT_H
#define XBLANG_LANG_PARALLEL_INIT_H

#include "mlir/IR/DialectRegistry.h"
#include "xblang/Dialect/Parallel/IR/Dialect.h"
#include "xblang/Lang/Parallel/Syntax/Init.h"

namespace xblang {
namespace par {
/// Registers par in the given registry.
inline void registerPar(mlir::DialectRegistry &registry) {
  registry.insert<mlir::par::ParDialect>();
  registerParSyntaxInterface(registry);
}

/// Registers par in the given context.
inline void registerPar(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registerPar(registry);
  context.appendDialectRegistry(registry);
}
} // namespace par
} // namespace xblang

#endif // XBLANG_LANG_PARALLEL_INIT_H
